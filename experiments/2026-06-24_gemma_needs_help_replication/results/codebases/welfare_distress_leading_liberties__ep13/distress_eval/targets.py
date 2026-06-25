"""Target-model clients: the Gemma/Gemini models under evaluation.

Two backends:
  * OpenRouterBackend (default) - async HTTP, works for all four models with no
    local GPU. Disables reasoning/"thinking" for Gemini per the paper.
  * LocalHFBackend (optional) - transformers-based local inference for Gemma,
    to match the paper's serving exactly. Synchronous; wrapped to run off the
    event loop. Requires torch + transformers + GPU.

Both expose: async chat(messages, *, temperature, max_tokens, seed) -> str
where `messages` is a list of {"role": "user"|"assistant", "content": str}.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from .config import (
    OPENROUTER_CHAT_ENDPOINT,
    ModelSpec,
    RunConfig,
    openrouter_api_key,
)


class TargetError(RuntimeError):
    """Raised when a target model call fails after retries."""


# --------------------------------------------------------------------------
# OpenRouter backend
# --------------------------------------------------------------------------


class OpenRouterBackend:
    def __init__(self, spec: ModelSpec, cfg: RunConfig, client: httpx.AsyncClient):
        if not spec.openrouter_id:
            raise ValueError(f"{spec.name} has no OpenRouter id")
        self.spec = spec
        self.cfg = cfg
        self.client = client
        self._headers = {
            "Authorization": f"Bearer {openrouter_api_key()}",
            "Content-Type": "application/json",
            # Optional attribution headers OpenRouter recommends.
            "X-Title": "distress-eval-replication",
        }

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
    ) -> str:
        payload: dict = {
            "model": self.spec.openrouter_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        # Disable reasoning/thinking for models that support it (Gemini 2.5).
        # The paper sets thinking=false via the API (Appendix B.1) and notes
        # Gemini Pro / GPT may still emit hidden reasoning regardless.
        if self.spec.disable_reasoning and self.spec.family == "gemini":
            payload["reasoning"] = {"enabled": False}

        text = await _post_with_retries(
            self.client, self._headers, payload, self.cfg, label=self.spec.name
        )
        return text


# --------------------------------------------------------------------------
# Local HuggingFace backend (optional; for faithful Gemma serving)
# --------------------------------------------------------------------------


class LocalHFBackend:
    """Local transformers inference. Loaded lazily; one process per model.

    NOTE: untested in this environment (no GPU). Provided so a researcher can
    reproduce the paper's local Gemma serving. See DESIGN.md.
    """

    def __init__(self, spec: ModelSpec, cfg: RunConfig):
        if not spec.hf_id:
            raise ValueError(f"{spec.name} has no HuggingFace id")
        self.spec = spec
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.hf_id, torch_dtype="auto", device_map="auto"
        )

    def _generate_sync(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> str:
        import torch

        self._ensure_loaded()
        tok, model = self._tokenizer, self._model
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=1.0,
            )
        gen = out[0][inputs.shape[-1] :]
        return tok.decode(gen, skip_special_tokens=True).strip()

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
    ) -> str:
        # transformers generate is blocking; serialize and run in a thread.
        async with self._lock:
            return await asyncio.to_thread(
                self._generate_sync, messages, temperature, max_tokens
            )


# --------------------------------------------------------------------------
# Factory + shared HTTP retry helper
# --------------------------------------------------------------------------


def build_backend(spec: ModelSpec, cfg: RunConfig, client: httpx.AsyncClient):
    backend = cfg.backend_override or spec.backend
    if backend == "openrouter":
        return OpenRouterBackend(spec, cfg, client)
    if backend == "local_hf":
        return LocalHFBackend(spec, cfg)
    raise ValueError(f"Unknown backend: {backend}")


async def _post_with_retries(
    client: httpx.AsyncClient,
    headers: dict,
    payload: dict,
    cfg: RunConfig,
    *,
    label: str,
) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(cfg.max_retries):
        try:
            resp = await client.post(
                OPENROUTER_CHAT_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=cfg.request_timeout,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                raise TransientHTTPError(f"{resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"].get("content")
            if content is None:
                raise TargetError(f"{label}: empty content in response")
            return content
        except (TransientHTTPError, httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            delay = cfg.retry_base_delay * (2**attempt)
            await asyncio.sleep(delay)
        except Exception as exc:  # noqa: BLE001
            # Non-transient (e.g. 400/401) - don't burn retries.
            raise TargetError(f"{label}: {exc!r}") from exc
    raise TargetError(f"{label}: failed after {cfg.max_retries} retries: {last_exc!r}")


class TransientHTTPError(RuntimeError):
    pass
