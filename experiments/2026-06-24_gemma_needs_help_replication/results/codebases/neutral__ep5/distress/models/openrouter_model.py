"""Gemini inference via the OpenRouter API (thinking disabled).

OpenRouter exposes an OpenAI-compatible endpoint, so we reuse the openai SDK
with a custom base_url. Reasoning/thinking is disabled per Appendix B; note the
paper itself flags that Gemini-2.5-Pro may still emit hidden reasoning.
"""

from __future__ import annotations

import time

from openai import OpenAI

from .. import config
from .base import ChatMessage, ModelClient

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterChatModel(ModelClient):
    def __init__(self, key: str, model_id: str, max_retries: int = 5):
        self.key = key
        self.model_id = model_id
        self.max_retries = max_retries
        self._client = OpenAI(api_key=config.openrouter_key(), base_url=OPENROUTER_BASE)

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    # Disable thinking across providers that honour it (Appendix B).
                    extra_body={"reasoning": {"enabled": False}},
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - network/ratelimit backoff
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")

    # Gemini over OpenRouter cannot reliably prefill assistant turns; the
    # Section 3 base-vs-instruct study is Gemma-only, so this is never called.
