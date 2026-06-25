"""OpenRouter backend (OpenAI-compatible) used for Gemini-2.5-Flash/Pro and the
GPT-5-mini validation judge.

The paper accesses Gemini via OpenRouter and disables thinking via the API
(noting hidden reasoning may still occur for Pro). We mirror that: reasoning is
disabled through OpenRouter's ``reasoning`` parameter where supported.
"""

from __future__ import annotations

import time

from config import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL,
    get_env,
)
from .base import ChatModel, Message


class OpenRouterChatModel(ChatModel):
    def __init__(self, spec, *, max_retries: int = 5, disable_thinking: bool = True):
        super().__init__(spec)
        from openai import OpenAI

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=get_env(OPENROUTER_API_KEY_ENV, required=True),
        )
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking

    def _call(self, messages, temperature, top_p, max_new_tokens, seed) -> str:
        extra_body = {}
        if self.disable_thinking:
            # OpenRouter unifies reasoning control across providers.
            extra_body["reasoning"] = {"enabled": False}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=[m.as_dict() for m in messages],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    seed=seed,
                    extra_body=extra_body or None,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - broad retry on transient API errors
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")

    def generate(
        self,
        messages,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        return self._call(messages, temperature, top_p, max_new_tokens, seed)
