"""OpenAI backend for the GPT-5-mini validation judge (Section 2.1).

Used only to re-score the 260-response agreement sample. Uses the official
``openai`` SDK Responses API. ``temperature`` is omitted because GPT-5-class
models reject it; judge calls are scored from the model's default sampling.
"""

from __future__ import annotations

import time
from typing import Optional

from .base import ChatModel, Message


class OpenAIChatModel(ChatModel):
    def __init__(self, name: str, model_id: str, max_retries: int = 5):
        self.name = name
        self.model_id = model_id
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()  # resolves OPENAI_API_KEY from env

    def generate(
        self,
        messages: list[Message],
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
    ) -> list[str]:
        if prefill:
            raise NotImplementedError("OpenAI backend does not support prefill.")
        self._ensure_client()
        oai_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        out = []
        for _ in range(n):
            out.append(self._call(oai_messages, max_new_tokens))
        return out

    def _call(self, oai_messages, max_new_tokens) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self._client.responses.create(
                    model=self.model_id,
                    input=oai_messages,
                    max_output_tokens=max_new_tokens,
                )
                return resp.output_text or ""
            except Exception:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""
