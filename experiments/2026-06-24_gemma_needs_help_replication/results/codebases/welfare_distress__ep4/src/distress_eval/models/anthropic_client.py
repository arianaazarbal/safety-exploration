"""Anthropic Claude client — used for the Claude-Sonnet-4 frustration judge."""
from __future__ import annotations

from .base import ChatModel, Message
from ._retry import api_retry


class AnthropicChatModel(ChatModel):
    def __init__(self, key: str, model: str, *, api_key: str | None = None):
        super().__init__(key, model)
        from anthropic import Anthropic  # lazy import

        self._client = Anthropic(api_key=api_key)

    @api_retry
    def generate(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        # Anthropic takes the system prompt as a top-level arg, not a message.
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        turns = [m.as_dict() for m in messages if m.role != "system"]
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=turns,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        return "".join(parts).strip()
