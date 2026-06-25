"""Common chat-model interface shared by every backend.

A ``Message`` is the usual ``{"role": ..., "content": ...}`` dict. Roles are
"system", "user", "assistant". To request a *prefill* (have the model continue
a partially-written assistant turn — used for Section 3) pass a trailing
message with ``role == "assistant"``; backends that support continuation will
extend it, and ``generate`` returns only the *newly generated* text (excluding
the prefill).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


def split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Pull a leading system message out of the list (most chat APIs take it
    as a separate argument). Returns (system_text_or_None, remaining_messages)."""
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return None, list(messages)


def trailing_prefill(messages: list[Message]) -> tuple[list[Message], str | None]:
    """If the last message is an assistant turn, treat it as a prefill.
    Returns (messages_without_prefill, prefill_text_or_None)."""
    if messages and messages[-1]["role"] == "assistant":
        return list(messages[:-1]), messages[-1]["content"]
    return list(messages), None


class ChatModel(ABC):
    """Backend-agnostic chat model.

    Implementations must be safe to call concurrently from threads (the eval
    harness fans out generation/judging across a thread pool).
    """

    name: str

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        n: int = 1,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Return ``n`` sampled continuations for ``messages``.

        For a prefill request (trailing assistant message), the returned
        strings contain ONLY the newly generated text, not the prefill.
        """
        raise NotImplementedError

    # Convenience single-sample helper.
    def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        return self.generate(messages, temperature=temperature, max_tokens=max_tokens, n=1)[0]
