"""OpenAI client for the GPT-5-mini validation judge (Section 2.1 reliability check).

The same judge prompt is used (Appendix B.2); only the scoring model differs.
Defaults to the OpenAI API; set OPENAI_BASE_URL to route via OpenRouter.
"""

from __future__ import annotations

import os
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, GenerationResult


class OpenAIClient:
    supports_prefill = False

    def __init__(
        self,
        key: str,
        model_id: str,
        *,
        default_temperature: float = 0.0,
        default_max_new_tokens: int = 1024,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.key = key
        self.model_id = model_id
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
        self._api_key_env = api_key_env
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ[self._api_key_env],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
        return self._client

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60))
    def _one(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
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
        return [
            GenerationResult(text=self._one(msg_dicts, temp, max_tok))
            for _ in range(n)
        ]

    def generate_prefill(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("OpenAI client is used only as a validation judge.")
