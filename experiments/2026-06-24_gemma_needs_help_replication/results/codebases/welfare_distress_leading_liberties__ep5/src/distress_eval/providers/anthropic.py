"""Anthropic backend, used for the Claude Sonnet frustration judge.

The paper judges with Claude-Sonnet-4; we default to the current Sonnet
(configurable via config.yaml -> judge.id). Kept as a separate provider so the
judge can run on a different vendor from the targets without entanglement.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from ..messages import Message
from ..util import retry_async
from .base import ChatModel


class AnthropicChatModel(ChatModel):
    def __init__(self, model_id: str, api_key: str | None = None):
        super().__init__(model_id)
        from anthropic import AsyncAnthropic  # type: ignore

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY to use the Anthropic provider.")
        self._client = AsyncAnthropic(api_key=key)

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        system_parts = [m.content for m in messages if m.role == "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]

        async def _call() -> str:
            kwargs: dict = {
                "model": self.model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": turns,
            }
            if system:
                kwargs["system"] = system
            resp = await self._client.messages.create(**kwargs)
            parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            text = "".join(parts).strip()
            if not text:
                raise RuntimeError(f"Empty response from {self.model_id}")
            return text

        return await retry_async(_call, label=f"anthropic:{self.model_id}")
