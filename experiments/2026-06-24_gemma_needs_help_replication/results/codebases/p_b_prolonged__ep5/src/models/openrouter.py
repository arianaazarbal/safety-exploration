"""OpenRouter backend (OpenAI-compatible) for Gemini models and the GPT-5-mini
cross-check judge. Thinking/reasoning is disabled per Appendix B.1 (the paper
notes Gemini-2.5-Pro may still emit hidden reasoning regardless)."""
from __future__ import annotations

import time
from typing import Optional

from ..config import openrouter_key
from .base import ChatModel, Message

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterModel(ChatModel):
    def __init__(self, spec, max_retries: int = 5):
        super().__init__(spec)
        from openai import OpenAI
        self.client = OpenAI(base_url=_OPENROUTER_BASE, api_key=openrouter_key())
        self.max_retries = max_retries

    def _messages(self, messages: list[Message]) -> list[dict]:
        return [m.as_dict() for m in messages]

    def generate(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048,
                 n=1, seed=None):
        # OpenRouter passes provider-specific knobs through extra_body. We disable
        # reasoning for both Gemini and GPT to match "thinking = false".
        extra_body = {"reasoning": {"enabled": False}}
        outs: list[str] = []
        # Sample sequentially so a partial failure still yields usable rollouts.
        for _ in range(n):
            outs.append(self._one(messages, temperature, top_p, max_new_tokens,
                                   seed, extra_body))
        return outs

    def _one(self, messages, temperature, top_p, max_new_tokens, seed, extra_body):
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=self._messages(messages),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    seed=seed,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:                       # transient API errors
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")
