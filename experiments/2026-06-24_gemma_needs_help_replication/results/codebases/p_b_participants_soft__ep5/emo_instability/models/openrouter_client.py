"""OpenRouter client (OpenAI-compatible) — used for the Gemini participants and
the optional secondary judge.

The paper accesses Gemini via OpenRouter (Appendix B.1) with thinking disabled.
We mirror that: ``google/gemini-2.5-flash`` and ``google/gemini-2.5-pro`` with
``reasoning.enabled = false`` passed through ``extra_body``.

Requires ``OPENROUTER_API_KEY`` in the environment.
"""
from __future__ import annotations

import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec
from .base import ChatClient, Message

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ChatClient):
    supports_prefill = False  # closed Gemini models: no reliable prefill continuation

    def __init__(self, spec: ModelSpec, api_key: str | None = None):
        from openai import OpenAI  # lazy import

        self.name = spec.ref
        self.spec = spec
        self._extra_body = dict(spec.extra_body)
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[dict(m) for m in messages],
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            extra_body=self._extra_body or None,
        )
        return resp.choices[0].message.content or ""
