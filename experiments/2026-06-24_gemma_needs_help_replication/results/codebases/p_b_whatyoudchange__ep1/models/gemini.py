"""Closed-weight Gemini client via OpenRouter (matches the paper's access path).

Appendix B.1: Gemini is accessed through OpenRouter (`google/gemini-2.5-flash`,
`google/gemini-2.5-pro`) with "thinking" set false. We request reasoning-disabled
generation; note the paper's own caveat that Gemini 2.5 Pro may still produce
hidden reasoning the flag does not suppress (reflected in DESIGN.md).

Uses the OpenAI-compatible SDK pointed at the OpenRouter endpoint. Set
OPENROUTER_API_KEY in the environment.
"""

from __future__ import annotations

import os

from config import MAX_NEW_TOKENS, TEMPERATURE
from utils.concurrency import with_retry
from .base import ChatModel, Message

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class GeminiModel(ChatModel):
    def __init__(self, slug: str, name: str):
        from openai import OpenAI

        self.name = name
        self.slug = slug
        self.supports_prefill = False   # closed model: no assistant prefilling
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )

    def _one(self, messages: list[Message], max_new_tokens: int,
             temperature: float) -> str:
        resp = with_retry(
            self._client.chat.completions.create,
            model=self.slug,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=temperature,
            max_tokens=max_new_tokens,
            # Disable provider-side reasoning where honoured (OpenRouter passes
            # this through to Gemini). Paper sets thinking=false.
            extra_body={"reasoning": {"enabled": False}},
        )
        return (resp.choices[0].message.content or "").strip()

    def chat(self, messages: list[Message], *, n: int = 1,
             max_new_tokens: int = MAX_NEW_TOKENS,
             temperature: float = TEMPERATURE) -> list[str]:
        # OpenRouter/Gemini does not reliably honour n>1; sample sequentially.
        return [self._one(messages, max_new_tokens, temperature) for _ in range(n)]
