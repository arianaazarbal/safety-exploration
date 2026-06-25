"""OpenRouter (OpenAI-compatible) backend.

Used for the closed-weight Gemini models (`google/gemini-2.5-flash`,
`google/gemini-2.5-pro`) -- the same access path the paper used (Appendix B.1) --
and for the GPT-5-mini reliability judge.

Requires `OPENROUTER_API_KEY`.
"""
from __future__ import annotations

import os
import time
from typing import Sequence

from .base import ChatMessage, GenerationResult, ModelBackend

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        *,
        family: str = "gemini",
        kind: str = "instruct",
        disable_thinking: bool = True,
        max_retries: int = 6,
    ):
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.client = OpenAI(base_url=_OPENROUTER_BASE, api_key=api_key)
        self.model_id = model_id
        self.family = family
        self.kind = kind
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[GenerationResult]:
        if prefill:
            # Assistant prefill / continuation is only needed for the Gemma-only
            # Section 3 experiment, and chat APIs don't continue assistant turns
            # reliably. We surface this rather than silently degrade.
            raise NotImplementedError(
                "assistant prefill is not supported on the OpenRouter backend "
                "(only used for Gemma base/instruct continuation in Section 3)"
            )

        extra_body: dict = {}
        if self.disable_thinking:
            # Paper sets "thinking=false" via the API (Appendix B.1). On
            # OpenRouter this maps to disabling the reasoning budget. Note the
            # paper's caveat that Gemini-2.5-Pro may still emit hidden reasoning.
            extra_body["reasoning"] = {"enabled": False}

        results: list[GenerationResult] = []
        for _ in range(n):
            text = self._one_call(
                list(messages), temperature, max_new_tokens, stop, extra_body
            )
            results.append(GenerationResult(text=text))
        return results

    def _one_call(self, messages, temperature, max_new_tokens, stop, extra_body):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    stop=list(stop) if stop else None,
                    extra_body=extra_body or None,
                )
                choice = resp.choices[0]
                return choice.message.content or ""
            except Exception as e:  # noqa: BLE001 -- transient API errors
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")
