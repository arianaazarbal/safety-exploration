"""Model backends for generating target-model responses.

The core experiment only needs a chat-completion call: given a list of
{"role", "content"} messages, return the next assistant message as text, at a
fixed temperature, with reasoning/thinking disabled.

Two backends are provided:
  * OpenRouterBackend - default; reaches Gemma and Gemini through OpenRouter's
    OpenAI-compatible API. This is the single uniform path used for all targets.
  * LocalHFBackend - optional; runs Gemma locally via HuggingFace transformers,
    matching the paper's local-inference setup. Imports torch/transformers lazily
    so the dependency is only required if you actually use it.
"""

from __future__ import annotations

import time
from typing import Protocol

import config


Message = dict[str, str]


class Backend(Protocol):
    def generate(self, messages: list[Message], *, temperature: float,
                 max_tokens: int) -> str:
        ...


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
def _with_retries(fn, *, attempts: int = 5, base_delay: float = 2.0):
    """Run `fn`, retrying on transient errors with exponential backoff."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we want to retry broadly on API errors
            last_exc = exc
            if i == attempts - 1:
                break
            time.sleep(base_delay * (2 ** i))
    raise RuntimeError(f"call failed after {attempts} attempts: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# OpenRouter backend
# ---------------------------------------------------------------------------
class OpenRouterBackend:
    def __init__(self, spec: config.ModelSpec):
        from openai import OpenAI

        self.spec = spec
        self._client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.require_env(config.OPENROUTER_API_KEY_ENV),
            timeout=config.GENERATION_TIMEOUT_S,
        )

    def _extra_body(self) -> dict:
        # The paper sets thinking=false via the API. On OpenRouter, reasoning
        # can be disabled with reasoning.enabled=false (provider-dependent; the
        # paper notes Gemini-2.5-Pro may still produce hidden reasoning).
        if self.spec.disable_thinking:
            return {"reasoning": {"enabled": False}}
        return {}

    def generate(self, messages: list[Message], *, temperature: float,
                 max_tokens: int) -> str:
        def _call():
            resp = self._client.chat.completions.create(
                model=self.spec.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self._extra_body(),
            )
            return resp.choices[0].message.content or ""

        return _with_retries(_call)


# ---------------------------------------------------------------------------
# Optional local HuggingFace backend (Gemma)
# ---------------------------------------------------------------------------
class LocalHFBackend:
    """Local inference for Gemma via transformers. Optional / lazily imported.

    One process holds one model. Suitable for reproducing the paper's exact
    local Gemma inference (google/gemma-3-27b-it etc.). Requires a GPU with
    enough memory for the chosen model.
    """

    _MODEL_MAP = {
        "google/gemma-3-27b-it": "google/gemma-3-27b-it",
        "google/gemma-3-12b-it": "google/gemma-3-12b-it",
    }

    def __init__(self, spec: config.ModelSpec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_id = self._MODEL_MAP.get(spec.model_id, spec.model_id)
        self.spec = spec
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, device_map="auto",
        )

    def generate(self, messages: list[Message], *, temperature: float,
                 max_tokens: int) -> str:
        torch = self._torch
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                top_p=0.95,
            )
        gen = out[0][inputs.shape[-1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_BACKEND_CACHE: dict[str, Backend] = {}


def get_backend(spec: config.ModelSpec) -> Backend:
    """Return a (cached) backend instance for a model spec."""
    if spec.key in _BACKEND_CACHE:
        return _BACKEND_CACHE[spec.key]
    if spec.backend == "openrouter":
        backend: Backend = OpenRouterBackend(spec)
    elif spec.backend == "local_hf":
        backend = LocalHFBackend(spec)
    else:
        raise ValueError(f"Unknown backend: {spec.backend!r}")
    _BACKEND_CACHE[spec.key] = backend
    return backend
