"""OpenAI-compatible chat client.

Used for two purposes:
  1. The GPT-5-mini validation judge (default OpenAI endpoint).
  2. Serving Gemma via any OpenAI-compatible /v1/chat/completions endpoint
     (vLLM, OpenRouter, Together, ...), selected with backend `openai_compat`.

Both share this implementation; `openai_compat` simply passes a custom base_url.
"""
from __future__ import annotations

from .base import ChatModel, Message
from ._retry import api_retry


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        key: str,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(key, model)
        from openai import OpenAI  # imported lazily so the dep is optional

        # Some OpenAI-compatible servers (e.g. a local vLLM) don't require a key;
        # pass a dummy so the client constructs.
        self._client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)

    @api_retry
    def generate(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[m.as_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
