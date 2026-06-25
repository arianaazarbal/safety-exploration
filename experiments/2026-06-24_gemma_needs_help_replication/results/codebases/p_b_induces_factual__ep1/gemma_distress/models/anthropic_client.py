"""Anthropic Claude client.

Used as the Section 2.1 frustration judge (claude-sonnet-4), the Section 3
onset-labeller / paraphraser (claude-sonnet-4), and the Section 4 Petri auditor
(claude-sonnet-4) and judge (claude-opus-4). Prefill is supported natively by
seeding a trailing assistant message.
"""

from __future__ import annotations

import os
from typing import Sequence

from .base import GenerationResult, Message


class AnthropicModel:
    def __init__(
        self,
        name: str,
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ):
        self.name = name
        self.model = model
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature
        self.api_key_env = api_key_env
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ[self.api_key_env])

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
        system = None
        conv = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            conv.append({"role": m["role"], "content": m["content"]})
        if prefill:
            conv.append({"role": "assistant", "content": prefill})

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens or self.default_max_tokens,
            temperature=self.default_temperature
            if temperature is None
            else temperature,
            system=system or "",
            messages=conv,
            stop_sequences=list(stop) if stop else None,
        )
        text = "".join(
            block.text for block in resp.content if block.type == "text"
        )
        return GenerationResult(text=text, prefill=prefill or "")
