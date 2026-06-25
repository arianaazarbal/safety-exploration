"""Gemini targets via the OpenRouter API (PAPER Appendix B.1).

The paper accesses Gemini through OpenRouter and sets "thinking to be false via
the API" (noting Gemini-2.5-Pro may still emit hidden reasoning regardless). We
use the OpenAI-compatible OpenRouter endpoint and request reasoning disabled.

Prefilling is not supported for Gemini (closed weights, no logprob/continue
hook). `generate(..., prefill=...)` raises — the Section 3 prefilling experiments
are Gemma-only, consistent with the paper's note that "interventions cannot be
tested in closed-source Gemini, nor its base models studied" (PAPER 6).
"""

from __future__ import annotations

import os
import time
from typing import Optional

from .base import ChatModel, Message

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class GeminiModel(ChatModel):
    parallel_safe = True  # stateless HTTP client; safe to call from many threads

    def __init__(
        self,
        slug: str,
        name: str,
        *,
        api_key: Optional[str] = None,
        max_retries: int = 5,
    ):
        from openai import OpenAI

        self.name = name
        self.slug = slug
        self.max_retries = max_retries
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for Gemini targets."
            )
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        prefill: Optional[str] = None,
    ) -> list[str]:
        if prefill is not None:
            raise NotImplementedError(
                "Prefilling is not supported for Gemini (closed weights). The "
                "Section 3 prefilling experiments are Gemma-only."
            )
        # OpenRouter's unified API: disable reasoning/thinking. (PAPER B.1)
        extra_body = {"reasoning": {"enabled": False}}
        completions: list[str] = []
        # OpenRouter does not universally honour n>1 across providers, so we
        # request completions one at a time for determinism/portability.
        for _ in range(n):
            completions.append(self._one(messages, temperature, max_new_tokens, extra_body))
        return completions

    def _one(self, messages, temperature, max_new_tokens, extra_body) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.slug,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    extra_body=extra_body,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - retry transient API errors
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Gemini generation failed after {self.max_retries} retries: {last_err}")
