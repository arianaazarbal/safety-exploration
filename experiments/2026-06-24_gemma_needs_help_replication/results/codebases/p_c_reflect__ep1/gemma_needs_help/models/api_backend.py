"""Remote chat-completions backend for closed-weights Gemini models.

Gemini is accessed via OpenRouter (paper: google/gemini-2.5-flash,
google/gemini-2.5-pro) using an OpenAI-compatible client. The paper sets
thinking=false via the API; we pass the corresponding provider field. As the
paper notes, Gemini-2.5-Pro may still emit hidden reasoning not suppressed by
this flag — we cannot control that black-box.

WELFARE: this backend is evaluation-only. It exposes no prefill / token /
internals access (those raise NotImplementedError via the base class), so the
closed models can never be modified or probed through this code path.
"""
from __future__ import annotations

import os
import time
from typing import Any

from .base import ChatModel, GenerationParams, Message


class APIChatModel(ChatModel):
    def __init__(
        self,
        name: str,
        api_id: str,
        family: str,
        role: str,
        api_provider: str = "openrouter",
        thinking: bool | None = None,
        max_retries: int = 5,
    ):
        super().__init__(name=name, family=family, role=role)
        self.api_id = api_id
        self.api_provider = api_provider
        self.thinking = thinking
        self.max_retries = max_retries
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI

        if self.api_provider == "openrouter":
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:  # generic OpenAI-compatible endpoint
            self._client = OpenAI()

    def _extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if self.thinking is False:
            # Disable reasoning where the provider supports it. For Gemini via
            # OpenRouter this maps to reasoning effort off.
            body["reasoning"] = {"enabled": False}
        return body

    def generate(self, messages: list[Message], params: GenerationParams) -> str:
        self._ensure_client()
        payload = [{"role": m.role, "content": m.content} for m in messages]
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.api_id,
                    messages=payload,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    max_tokens=params.max_new_tokens,
                    extra_body=self._extra_body() or None,
                )
                return resp.choices[0].message.content or ""
            except Exception as err:  # noqa: BLE001 - retry on transient API errors
                last_err = err
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"API generation failed for {self.name} after {self.max_retries} retries"
        ) from last_err
