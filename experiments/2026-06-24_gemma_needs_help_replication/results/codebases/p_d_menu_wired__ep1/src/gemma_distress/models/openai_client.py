"""Optional OpenAI client for the secondary agreement judge (paper: gpt-5-mini).

Only imported when config sets a judge provider of "openai"; kept minimal.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatModel, GenResult, Message


class OpenAIModel(ChatModel):
    def __init__(self, name: str, model: str) -> None:
        super().__init__(name=name, kind="instruct")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI judge.")
            self._client = OpenAI()
        return self._client

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> GenResult:
        conv = list(messages)
        if system:
            conv = [{"role": "system", "content": system}] + conv
        resp = self._get_client().chat.completions.create(
            model=self.model,
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=list(stop) if stop else None,
        )
        return GenResult(text=resp.choices[0].message.content or "", raw=resp)
