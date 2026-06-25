"""OpenRouter backend for the Gemini participants (Appendix B.1).

Gemini-2.5-Flash / -Pro are accessed via OpenRouter's OpenAI-compatible
``/chat/completions`` endpoint, exactly as the paper does. Thinking is disabled
via the ``reasoning`` control where the route honours it (the paper notes Pro may
still emit hidden reasoning regardless).

Requires ``OPENROUTER_API_KEY`` in the environment. Sampling at temperature 1
(Section 2.1). API backends cannot do genuine assistant prefill continuation, so
:meth:`continue_from_prefill` is intentionally unsupported (Section 3 is
Gemma-only — see DESIGN.md).
"""

from __future__ import annotations

import os
import time

import requests

from .base import ChatModel, Message

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterChatModel(ChatModel):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        thinking: bool = False,
        api_key: str | None = None,
        max_retries: int = 5,
        timeout: float = 120.0,
    ):
        self.name = name
        self.model_id = model_id
        self.thinking = thinking
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for Gemini participants."
            )
        self.max_retries = max_retries
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[Message], temperature: float, max_new_tokens: int) -> dict:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        # OpenRouter exposes a unified `reasoning` control; disable to match
        # "we set thinking to be false via the API" (Appendix B.1).
        if not self.thinking:
            payload["reasoning"] = {"enabled": False}
        return payload

    def _request_once(self, payload: dict) -> str:
        resp = requests.post(
            OPENROUTER_URL, headers=self._headers(), json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[str]:
        payload = self._payload(messages, temperature, max_new_tokens)
        completions: list[str] = []
        # Sample sequentially: temperature-1 sampling gives independent draws,
        # and not all routes honour the `n` parameter reliably.
        for _ in range(n):
            completions.append(self._with_retries(payload))
        return completions

    def _with_retries(self, payload: dict) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._request_once(payload)
            except Exception as err:  # noqa: BLE001 - retry on any transient API error
                last_err = err
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"OpenRouter request failed after {self.max_retries} retries") from last_err
