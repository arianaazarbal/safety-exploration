"""Anthropic backend for the Claude judge, Petri auditor, and Petri judge.

These are measurement-infrastructure models, not subjects of the study. We use
the official ``anthropic`` SDK. Thinking is left off for the deterministic
scoring calls; the auditor is a plain multi-turn chat driver.
"""

from __future__ import annotations

import time
from typing import Optional

from .base import ChatModel, Message


class AnthropicChatModel(ChatModel):
    def __init__(self, name: str, model_id: str, max_retries: int = 5):
        self.name = name
        self.model_id = model_id
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            # Resolves ANTHROPIC_API_KEY from the environment.
            self._client = anthropic.Anthropic()

    @staticmethod
    def _split(messages: list[Message]):
        system = None
        conv = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            conv.append({"role": m["role"], "content": m["content"]})
        return system, conv

    def generate(
        self,
        messages: list[Message],
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
    ) -> list[str]:
        self._ensure_client()
        system, conv = self._split(messages)
        if prefill:
            conv = conv + [{"role": "assistant", "content": prefill}]
        out = []
        for _ in range(n):
            out.append(self._call(system, conv, temperature, max_new_tokens))
        return out

    def _call(self, system, conv, temperature, max_new_tokens) -> str:
        kwargs = dict(model=self.model_id, max_tokens=max_new_tokens, messages=conv)
        if system:
            kwargs["system"] = system
        # temperature is accepted on the Claude 4.x judge/auditor models used here.
        kwargs["temperature"] = temperature
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(**kwargs)
                return "".join(b.text for b in resp.content if b.type == "text")
            except Exception:  # noqa: BLE001 - backoff on transient API errors
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""
