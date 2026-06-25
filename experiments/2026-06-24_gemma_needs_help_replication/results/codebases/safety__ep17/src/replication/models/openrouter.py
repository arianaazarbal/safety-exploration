"""OpenRouter client for API models (Gemini, and the optional GPT-5-mini judge).

The paper accesses Gemini through OpenRouter (Appendix B.1), with thinking
disabled. We mirror that here using the OpenAI-compatible OpenRouter endpoint.
Reasoning/thinking is disabled via the ``reasoning`` extra-body field; note the
paper's caveat that Gemini-2.5-Pro may still emit hidden reasoning the flag does
not suppress.
"""
from __future__ import annotations

import os
import time

import config
from .base import Message, ModelClient

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(ModelClient):
    def __init__(self, spec: "config.ModelSpec", max_retries: int = 5):
        super().__init__(spec)
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"{config.OPENROUTER_API_KEY_ENV} not set; required for "
                f"API model '{self.spec.key}'."
            )
        self._client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=key)

    def chat(self, messages: list[Message], *, temperature=config.TEMPERATURE,
             max_new_tokens=config.MAX_NEW_TOKENS) -> str:
        self._ensure_client()
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    # Disable thinking/reasoning (paper sets thinking=false).
                    extra_body={"reasoning": {"enabled": False}},
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 -- transient API errors
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter call failed after {self.max_retries} tries: {last_err}")
