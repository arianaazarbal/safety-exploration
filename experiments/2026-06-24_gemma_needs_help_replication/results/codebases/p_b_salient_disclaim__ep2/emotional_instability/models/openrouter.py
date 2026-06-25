"""Gemini inference via OpenRouter (Appendix B.1).

The paper accesses Gemini-2.5-Flash and Gemini-2.5-Pro through OpenRouter with
thinking disabled. We use the OpenAI-compatible chat-completions client pointed
at the OpenRouter base URL.

Gemini does not support assistant prefill through this path and has no public
base model, so `supports_prefill` is False -- the Section 3 prefill comparison
is Gemma-only (see DESIGN.md).
"""

from __future__ import annotations

import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, GenerationResult

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    supports_prefill = False

    def __init__(
        self,
        key: str,
        model_id: str,
        *,
        default_temperature: float = 1.0,
        default_max_new_tokens: int = 2048,
        thinking: bool = False,
        api_key_env: str = "OPENROUTER_API_KEY",
    ):
        self.key = key
        self.model_id = model_id
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
        self.thinking = thinking
        self._api_key_env = api_key_env
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=os.environ[self._api_key_env],
            )
        return self._client

    def _reasoning_kwargs(self) -> dict:
        # OpenRouter exposes a `reasoning` control; disable thinking per B.1.
        # (Gemini-2.5-Pro may still produce hidden reasoning regardless.)
        if self.thinking:
            return {}
        return {"extra_body": {"reasoning": {"enabled": False}}}

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60))
    def _one(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **self._reasoning_kwargs(),
        )
        return resp.choices[0].message.content or ""

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        n: int = 1,
    ) -> list[GenerationResult]:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        temp = temperature if temperature is not None else self.default_temperature
        max_tok = max_new_tokens or self.default_max_new_tokens
        # Sample sequentially (temperature 1) -- providers vary in n support.
        return [
            GenerationResult(text=self._one(msg_dicts, temp, max_tok))
            for _ in range(n)
        ]

    def generate_prefill(self, *args, **kwargs):  # pragma: no cover - unsupported
        raise NotImplementedError(
            "Gemini via OpenRouter does not support assistant prefill; the "
            "Section 3 prefill comparison is Gemma-only (see DESIGN.md)."
        )
