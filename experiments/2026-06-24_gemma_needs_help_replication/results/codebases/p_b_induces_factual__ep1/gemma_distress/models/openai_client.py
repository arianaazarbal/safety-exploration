"""OpenAI client, used for the GPT-5-mini judge-validation re-scoring (Sec 2.1).

Works against the OpenAI API by default; point ``base_url`` at OpenRouter to use
the paper's access path.
"""

from __future__ import annotations

import os
from typing import Sequence

from .base import GenerationResult, Message


class OpenAIModel:
    def __init__(
        self,
        name: str,
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        n_validation: int | None = None,
    ):
        self.name = name
        self.model = model
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.n_validation = n_validation
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ[self.api_key_env], base_url=self.base_url
            )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        self._ensure_client()
        msgs = [dict(m) for m in messages]
        if prefill:
            msgs.append({"role": "assistant", "content": prefill})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.default_temperature
            if temperature is None
            else temperature,
            max_tokens=max_new_tokens or self.default_max_tokens,
            stop=list(stop) if stop else None,
        )
        text = resp.choices[0].message.content or ""
        return GenerationResult(text=text, prefill=prefill or "")
