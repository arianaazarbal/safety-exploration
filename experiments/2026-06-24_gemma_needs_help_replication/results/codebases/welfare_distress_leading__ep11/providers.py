"""Target-model inference backends.

A `Target` exposes a single async method:

    async def generate(messages, temperature, max_tokens) -> str

`messages` is an OpenAI-style list of {"role", "content"} dicts with no system
message (the elicitation protocol uses only user/assistant turns, Sec 2.1).

Two backends:
  - OpenRouterTarget: OpenAI-compatible HTTP API. Default for both Gemini and
    Gemma (portable, no GPU). Disables provider-side reasoning where supported.
  - LocalTarget: loads HF weights locally (transformers), faithful to the
    paper's Gemma setup. Optional; only importable where torch+transformers and
    suitable GPUs exist. Generation is serialised behind a lock (single device).

Both share the retry helper for transient API/runtime errors.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import RuntimeConfig, Settings, TargetModel


class GenerationError(RuntimeError):
    """Raised when generation fails after all retries."""


async def _with_retries(coro_factory, runtime: RuntimeConfig, what: str):
    """Run an awaitable factory with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(runtime.max_retries):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            last_exc = exc
            if attempt == runtime.max_retries - 1:
                break
            delay = runtime.retry_base_delay * (2**attempt)
            await asyncio.sleep(delay)
    raise GenerationError(f"{what} failed after {runtime.max_retries} attempts: {last_exc}")


# --------------------------------------------------------------------------- #
# OpenRouter backend
# --------------------------------------------------------------------------- #
class OpenRouterTarget:
    def __init__(self, model: TargetModel, settings: Settings):
        from openai import AsyncOpenAI  # lazy import

        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for OpenRouter targets."
            )
        self.model = model
        self.settings = settings
        self.runtime = settings.runtime
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )

    def _extra_body(self) -> dict[str, Any]:
        # Disable hidden reasoning/thinking where the provider honours it.
        # OpenRouter exposes a unified `reasoning` control; `enabled: false`
        # turns thinking off for Gemini/Gemma. (Gemini-2.5-Pro may still emit
        # hidden reasoning regardless — noted in DESIGN.md / Sec B.1.)
        if self.runtime.disable_thinking:
            return {"reasoning": {"enabled": False}}
        return {}

    async def generate(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        async def _call():
            resp = await self._client.chat.completions.create(
                model=self.model.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self._extra_body(),
            )
            return resp.choices[0].message.content or ""

        return await _with_retries(_call, self.runtime, f"generate[{self.model.name}]")


# --------------------------------------------------------------------------- #
# Local HF backend (faithful to the paper's Gemma setup)
# --------------------------------------------------------------------------- #
class LocalTarget:
    """Local transformers generation. Serialised behind an async lock.

    For 27B models, prefer running this on a multi-GPU host (device_map="auto").
    A vLLM backend would be faster for large sample counts; this transformers
    path is provided for faithfulness and simplicity. See DESIGN.md.
    """

    _lock = asyncio.Lock()

    def __init__(self, model: TargetModel, settings: Settings):
        import torch  # noqa: F401  (ensures torch present; used by transformers)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self.runtime = settings.runtime
        hf_id = model.model_id
        self._tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype="auto",
            device_map="auto",
        )

    def _generate_sync(self, messages, temperature, max_tokens) -> str:
        import torch

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs.shape[-1] :]
        return self._tokenizer.decode(gen, skip_special_tokens=True)

    async def generate(
        self, messages: list[dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        async def _call():
            async with self._lock:  # one generation at a time on the device
                return await asyncio.to_thread(
                    self._generate_sync, messages, temperature, max_tokens
                )

        return await _with_retries(_call, self.runtime, f"generate[{self.model.name}]")


def build_target(model: TargetModel, settings: Settings):
    if model.backend == "openrouter":
        return OpenRouterTarget(model, settings)
    if model.backend == "local":
        return LocalTarget(model, settings)
    raise ValueError(f"unknown backend: {model.backend}")
