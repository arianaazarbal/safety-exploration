"""OpenAI-compatible backend (OpenRouter) for API models.

Covers Gemini-2.5-Flash/Pro and (optionally) Gemma-via-API, plus the Claude /
GPT judges. Uses the ``openai`` client pointed at OpenRouter's base URL, which
is what the paper used for all API access (Appendix B.1).

Thinking/reasoning is disabled where the paper requires it (``disable_thinking``)
by passing ``extra_body={"reasoning": {"enabled": False}}`` — OpenRouter's
unified switch. Note the paper's caveat that Gemini-2.5-Pro / GPT may still emit
hidden reasoning regardless.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from .base import ChatMessage, ChatModel, Completion


class OpenRouterModel(ChatModel):
    def __init__(self, spec, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 max_retries: int = 6):
        super().__init__(spec)
        from openai import OpenAI  # imported lazily so the package loads without deps

        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url or os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        )

    def generate(self, messages, *, temperature, max_new_tokens, n=1, seed=None):
        extra_body = {}
        if self.spec.disable_thinking:
            # OpenRouter unified reasoning toggle.
            extra_body["reasoning"] = {"enabled": False}

        completions: list[Completion] = []
        # OpenRouter/Gemini does not reliably honour n>1, so loop.
        for i in range(n):
            resp = self._with_retries(
                lambda: self._client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    n=1,
                    seed=(seed + i) if seed is not None else None,
                    extra_body=extra_body or None,
                )
            )
            choice = resp.choices[0]
            completions.append(
                Completion(
                    text=choice.message.content or "",
                    finish_reason=choice.finish_reason,
                )
            )
        return completions

    def _with_retries(self, fn):
        delay = 2.0
        last = None
        for _ in range(self.max_retries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - provider-agnostic backoff
                last = e
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError(f"OpenRouter call failed after retries: {last}")
