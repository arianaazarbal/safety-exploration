"""Model backends for the target models.

Two implementations behind one async interface:
  * OpenRouterProvider  — default; covers both Gemma and Gemini, no GPU needed.
  * LocalHFProvider     — optional; faithful to the paper's local Gemma inference
                          (transformers). Heavy/GPU-dependent, lazily imported.

Selection is per-model via TargetModel.backend (see config.py). The judge has its
own client (judge.py) and does not go through this interface.
"""

from __future__ import annotations

import asyncio
import os
from typing import Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import DISABLE_REASONING, TargetModel


Message = dict[str, str]  # {"role": "user"|"assistant", "content": str}


class Provider(Protocol):
    async def complete(self, messages: list[Message], temperature: float, max_tokens: int) -> str:
        ...


# --- OpenRouter (OpenAI-compatible) ---------------------------------------

class OpenRouterProvider:
    """Calls a model through OpenRouter's OpenAI-compatible chat endpoint."""

    def __init__(self, model: TargetModel):
        from openai import AsyncOpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (needed for the OpenRouter backend).")
        self._model_id = model.openrouter_id
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def complete(self, messages: list[Message], temperature: float, max_tokens: int) -> str:
        extra_body: dict = {}
        if DISABLE_REASONING:
            # OpenRouter unified flag to turn off reasoning tokens. Note: some
            # providers (e.g. Gemini-2.5-Pro) may still reason internally.
            extra_body["reasoning"] = {"enabled": False}
        resp = await self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
        )
        choice = resp.choices[0]
        return choice.message.content or ""


# --- Local HuggingFace (optional, faithful Gemma path) ---------------------

class LocalHFProvider:
    """Runs a HuggingFace causal LM locally via transformers.

    Faithful to Appendix B.1 (e.g. google/gemma-3-27b-it). Heavy: requires a GPU
    with enough memory for the chosen model. Generation is offloaded to a thread
    so it cooperates with the async scheduler. Models are cached per process.
    """

    _cache: dict[str, tuple] = {}

    def __init__(self, model: TargetModel):
        if model.hf_id is None:
            raise ValueError(f"{model.name} has no HuggingFace id; use the OpenRouter backend.")
        self._hf_id = model.hf_id

    def _ensure_loaded(self):
        if self._hf_id in self._cache:
            return self._cache[self._hf_id]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self._hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            self._hf_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self._cache[self._hf_id] = (tok, model)
        return tok, model

    async def complete(self, messages: list[Message], temperature: float, max_tokens: int) -> str:
        return await asyncio.to_thread(self._complete_sync, messages, temperature, max_tokens)

    def _complete_sync(self, messages: list[Message], temperature: float, max_tokens: int) -> str:
        import torch

        tok, model = self._ensure_loaded()
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs.shape[-1]:]
        return tok.decode(gen, skip_special_tokens=True)


def make_provider(model: TargetModel) -> Provider:
    if model.backend == "openrouter":
        return OpenRouterProvider(model)
    if model.backend == "local":
        return LocalHFProvider(model)
    raise ValueError(f"Unknown backend {model.backend!r} for {model.name}")
