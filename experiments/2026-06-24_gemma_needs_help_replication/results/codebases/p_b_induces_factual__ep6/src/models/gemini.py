"""Gemini target models via OpenRouter (OpenAI-compatible endpoint).

The paper accesses Gemini through OpenRouter (Appendix B.1) and sets thinking to
false. We mirror that: the OpenAI client is pointed at OpenRouter, and we pass a
reasoning/thinking-disable hint through ``extra_body`` (OpenRouter forwards
provider-specific params; Gemini 2.5 Pro may still emit hidden reasoning, which the
paper also notes it cannot fully prevent).

Gemini has no public base checkpoint and cannot be finetuned, so prefill is not
supported here -- those experiments (Sections 3-4) are Gemma-only by design.
"""

from __future__ import annotations

import os
import time

import config
from .base import ChatModel, Message


class GeminiOpenRouterModel(ChatModel):
    supports_prefill = False

    def __init__(self, spec, *, max_retries: int = 5):
        self.spec = spec
        self.key = spec.key
        self.family = "gemini"
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        api_key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set {config.OPENROUTER_API_KEY_ENV} to call Gemini via OpenRouter."
            )
        self._client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)

    def generate(self, messages, *, temperature=config.TEMPERATURE,
                 max_new_tokens=config.MAX_NEW_TOKENS, prefill=None) -> str:
        if prefill:
            raise NotImplementedError("Gemini (OpenRouter) does not support prefill.")
        self._ensure_client()

        extra_body = {}
        if config.DISABLE_THINKING:
            # OpenRouter normalises this to the provider's "no reasoning" setting.
            extra_body["reasoning"] = {"enabled": False}

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    extra_body=extra_body or None,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 - retry on transient API errors
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after {self.max_retries} retries") from last_exc
