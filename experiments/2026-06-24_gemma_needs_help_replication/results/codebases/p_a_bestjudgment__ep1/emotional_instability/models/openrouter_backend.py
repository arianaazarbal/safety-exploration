"""Gemini inference via OpenRouter (Appendix B.1).

The paper accesses Gemini through OpenRouter slugs `google/gemini-2.5-flash` and
`google/gemini-2.5-pro`, and sets "thinking to be false via the API" (with the
caveat that Gemini-2.5-Pro may still produce hidden reasoning). We mirror that:
temperature 1, thinking disabled via OpenRouter's `reasoning` control.

OpenRouter exposes an OpenAI-compatible Chat Completions endpoint, so we use the
`openai` SDK pointed at OpenRouter's base URL. Requests are retried with
exponential backoff on transient errors.
"""

from __future__ import annotations

import os
import random
import time

from ..config import MAX_NEW_TOKENS, SAMPLING_TEMPERATURE, ModelSpec
from .base import Message, ModelBackend

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(ModelBackend):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def chat(self, messages: list[Message], *,
             temperature: float = SAMPLING_TEMPERATURE,
             max_new_tokens: int = MAX_NEW_TOKENS,
             n: int = 1) -> list[str]:
        self._ensure_client()
        extra_body = {}
        if self.spec.disable_thinking:
            # OpenRouter unifies provider reasoning controls under `reasoning`.
            # `enabled: false` disables thinking where the provider supports it.
            extra_body["reasoning"] = {"enabled": False}

        completions: list[str] = []
        # OpenRouter/Gemini does not reliably honour n>1, so loop.
        for _ in range(n):
            text = self._one_call(messages, temperature, max_new_tokens, extra_body)
            completions.append(text)
        return completions

    def _one_call(self, messages, temperature, max_new_tokens, extra_body,
                  max_retries: int = 6) -> str:
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    extra_body=extra_body or None,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - retry all transient API errs
                last_exc = exc
                delay = min(2 ** attempt + random.uniform(0, 1), 30)
                time.sleep(delay)
        raise RuntimeError(f"OpenRouter call failed after retries: {last_exc}")
