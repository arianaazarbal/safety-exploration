"""Target-model chat clients (Gemma + Gemini).

Two backends:
  * OpenRouter (default): one OpenAI-compatible client serves all four target
    models. This is how the paper accessed Gemini; Gemma 3 is also served here.
  * HuggingFace local (Gemma only): set GEMMA_BACKEND=huggingface to run Gemma
    locally with transformers, matching the paper's local-inference setup.

All clients expose `.chat(messages) -> str`, taking a list of
{"role": "user"|"assistant", "content": str} and returning the assistant text.
"""

from __future__ import annotations

import os
import time

import config
from config import TargetModel


class TargetClientError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# OpenRouter backend.
# ---------------------------------------------------------------------------

class OpenRouterClient:
    def __init__(self, model: TargetModel):
        from openai import OpenAI

        api_key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise TargetClientError(
                f"Set {config.OPENROUTER_API_KEY_ENV} to use the OpenRouter backend."
            )
        self.model = model
        self._client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)

    def chat(self, messages: list[dict]) -> str:
        kwargs: dict = dict(
            model=self.model.openrouter_id,
            messages=messages,
            temperature=config.TARGET_TEMPERATURE,
            max_tokens=config.TARGET_MAX_TOKENS,
        )
        # Best-effort thinking disable. OpenRouter accepts a `reasoning` field;
        # `enabled: False` disables reasoning where the provider supports it.
        # The paper notes Gemini-2.5-Pro may still emit hidden reasoning.
        if config.DISABLE_THINKING:
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}

        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                return content or ""
            except Exception as exc:  # noqa: BLE001 - retry on any transient API error
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise TargetClientError(f"OpenRouter call failed after retries: {last_exc}")


# ---------------------------------------------------------------------------
# HuggingFace local backend (Gemma only).
# ---------------------------------------------------------------------------

class HuggingFaceGemmaClient:
    # Cache loaded models across instances (loading a 27B model is expensive).
    _cache: dict[str, tuple] = {}

    def __init__(self, model: TargetModel):
        if model.hf_id is None:
            raise TargetClientError(f"{model.name} has no HuggingFace id (local backend).")
        self.model = model
        self._tokenizer, self._model = self._load(model.hf_id)

    @classmethod
    def _load(cls, hf_id: str):
        if hf_id in cls._cache:
            return cls._cache[hf_id]
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        cls._cache[hf_id] = (tok, model)
        return tok, model

    def chat(self, messages: list[dict]) -> str:
        import torch  # type: ignore

        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                inputs,
                max_new_tokens=config.TARGET_MAX_TOKENS,
                do_sample=True,
                temperature=config.TARGET_TEMPERATURE,
                top_p=1.0,
            )
        gen = out[0][inputs.shape[-1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------

def make_client(model: TargetModel):
    """Return the appropriate chat client for a target model."""
    if model.family == "gemma" and config.GEMMA_BACKEND == "huggingface":
        return HuggingFaceGemmaClient(model)
    return OpenRouterClient(model)
