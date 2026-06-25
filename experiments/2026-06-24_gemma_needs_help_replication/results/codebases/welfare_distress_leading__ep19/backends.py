"""Inference backends for generating model responses.

Two pluggable backends behind a common async ``generate`` interface:

  * OpenAICompatibleBackend  - any OpenAI-compatible chat endpoint. Used for
    OpenRouter (default) AND for a locally-served vLLM endpoint; only the
    base_url / model id differ. This is the default path.
  * TransformersBackend      - loads a HF model locally with `transformers` and
    applies its chat template. Most faithful to the paper's local Gemma runs,
    but requires GPUs and the heavy `torch`/`transformers` deps (imported lazily).

Both implement ``async generate(messages, *, temperature, max_tokens) -> str``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    OPENROUTER_BASE_URL,
    ModelSpec,
    get_openrouter_api_key,
)

Message = Dict[str, str]  # {"role": ..., "content": ...}


class Backend(Protocol):
    async def generate(
        self, messages: List[Message], *, temperature: float, max_tokens: int
    ) -> str: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (OpenRouter / local vLLM / any compatible server)
# ---------------------------------------------------------------------------

class OpenAICompatibleBackend:
    def __init__(
        self,
        model: ModelSpec,
        *,
        base_url: str = OPENROUTER_BASE_URL,
        api_key: Optional[str] = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        key = api_key or get_openrouter_api_key()
        if not key:
            raise RuntimeError(
                "No API key for OpenAI-compatible backend. Set OPENROUTER_API_KEY "
                "(or pass api_key / point base_url at a keyless local server)."
            )
        self._client = AsyncOpenAI(base_url=base_url, api_key=key)

    def _extra_body(self) -> Dict:
        # Disable hidden reasoning/thinking where the provider supports it.
        # NB: the paper notes Gemini-2.5-Pro may still emit hidden reasoning
        # that this flag does not fully suppress.
        if self.model.disable_thinking:
            return {"reasoning": {"enabled": False}}
        return {}

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    async def generate(
        self, messages: List[Message], *, temperature: float, max_tokens: int
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self._extra_body(),
        )
        choice = resp.choices[0]
        return (choice.message.content or "").strip()


# ---------------------------------------------------------------------------
# Local HuggingFace transformers backend (optional; GPU-heavy)
# ---------------------------------------------------------------------------

class TransformersBackend:
    """Local inference via `transformers`. Lazy, single-process, blocking
    generation wrapped to satisfy the async interface. Intended for the
    paper-faithful local Gemma path (google/gemma-3-27b-it etc.).
    """

    def __init__(self, model: ModelSpec, *, device_map: str = "auto", dtype: str = "bfloat16") -> None:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self._tokenizer = AutoTokenizer.from_pretrained(model.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model.model_id, device_map=device_map, torch_dtype=dtype
        )

    async def generate(
        self, messages: List[Message], *, temperature: float, max_tokens: int
    ) -> str:
        import asyncio

        # Run blocking generation in a thread so we don't stall the event loop.
        return await asyncio.to_thread(
            self._generate_sync, messages, temperature, max_tokens
        )

    def _generate_sync(
        self, messages: List[Message], temperature: float, max_tokens: int
    ) -> str:
        import torch

        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        gen = out[0][inputs.shape[-1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()


def make_backend(model: ModelSpec) -> Backend:
    if model.backend == "openai_compatible":
        return OpenAICompatibleBackend(model)
    if model.backend == "transformers":
        return TransformersBackend(model)
    raise ValueError(f"unknown backend: {model.backend}")
