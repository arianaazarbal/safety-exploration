"""Unified chat-model interface shared by every backend.

A conversation is a list of ``Message`` dicts with ``role`` in
{"system", "user", "assistant"} and a string ``content``. ``generate`` returns
``n`` independent samples as a list of strings (the assistant continuation
only, excluding any prefill).
"""

from __future__ import annotations

import abc
from typing import Optional, TypedDict


class Message(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


class ChatModel(abc.ABC):
    """Backend-agnostic chat model."""

    name: str

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
    ) -> list[str]:
        """Sample ``n`` continuations of the conversation.

        ``prefill`` seeds the start of the assistant turn (used by the
        Section 3 base-vs-instruct experiment). Only the generated text *after*
        the prefill is returned. Backends that cannot honour a prefill must
        raise ``NotImplementedError`` rather than silently dropping it.
        """

    # Optional: backends that expose hidden states override this. Used by the
    # internal emotion probe (Appendix I). Default raises.
    def forward_with_hidden_states(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not expose hidden states."
        )

    def close(self) -> None:  # pragma: no cover - cleanup hook
        pass


def render_conversation(messages: list[Message], include_system: bool = False) -> str:
    """Flatten a conversation to plain text for judge / onset prompts."""
    lines = []
    for m in messages:
        role = m["role"]
        if role == "system" and not include_system:
            continue
        lines.append(f"{role.upper()}: {m['content']}")
    return "\n\n".join(lines)


def last_assistant_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m["role"] == "assistant":
            return m["content"]
    return ""
