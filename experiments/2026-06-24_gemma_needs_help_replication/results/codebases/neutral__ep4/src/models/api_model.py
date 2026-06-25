"""OpenRouter (OpenAI-compatible) backend for Gemini targets.

Used for google/gemini-2.5-flash and google/gemini-2.5-pro. Reasoning/thinking
is disabled via the API where supported (the paper sets "thinking to be false",
noting Gemini-2.5-Pro may still produce hidden reasoning that this cannot fully
prevent).
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from config import OPENROUTER_API_KEY_ENV, OPENROUTER_BASE_URL
from .base import Message, ModelClient


class OpenRouterModel(ModelClient):
    def __init__(self, name: str, model_id: str):
        from openai import OpenAI

        self.name = name
        self.model_id = model_id
        api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"Set {OPENROUTER_API_KEY_ENV} to query {model_id}")
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def generate(self, messages, *, max_new_tokens=2048, temperature=1.0,
                 prefill=None) -> str:
        if prefill is not None:
            raise NotImplementedError(
                "Prefill is unsupported for closed-source Gemini "
                "(see paper limitation: Gemini base models cannot be studied).")

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            # Disable thinking/reasoning where the provider honours it.
            extra_body={"reasoning": {"enabled": False}},
        )
        return (resp.choices[0].message.content or "").strip()
