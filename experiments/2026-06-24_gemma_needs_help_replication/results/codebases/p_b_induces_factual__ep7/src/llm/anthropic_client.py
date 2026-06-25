"""Anthropic client used for the Claude-based roles: frustration judge (Sec 2.1),
onset labeller and paraphraser (Sec 3 / App C), Petri auditor and Petri judge (App G).

Reads ``ANTHROPIC_API_KEY``. Exposes a single ``complete`` method that takes a system
string plus a normalised message list and returns text — sufficient for both
single-shot judging and multi-turn auditing.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import Message, ChatModel
from config import MAX_API_RETRIES, API_BACKOFF_BASE


class AnthropicClient:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY for Claude judge/auditor access")
        self._client = anthropic.Anthropic()

    def complete(
        self,
        messages: list[Message],
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        self._ensure_client()
        # Pull any system messages out of the list (Anthropic takes system separately).
        sys_parts = [m["content"] for m in messages if m["role"] == "system"]
        if system:
            sys_parts.insert(0, system)
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]

        def _call():
            resp = self._client.messages.create(
                model=self.model_id,
                system="\n\n".join(sys_parts) if sys_parts else None,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return "".join(block.text for block in resp.content if block.type == "text").strip()

        return ChatModel._retry(_call, retries=MAX_API_RETRIES, base=API_BACKOFF_BASE)
