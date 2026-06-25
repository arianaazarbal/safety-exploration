"""Shared chat client interface.

A `ChatClient` turns a list of chat messages into an assistant completion. The
critical capability for this paper is `prefill=`: forcing the assistant turn to
begin with a fixed string (used by Section 3's truncation experiment and to make
base models continue chat-formatted conversations). Only local HF models support
prefill; API backends raise if asked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@runtime_checkable
class ChatClient(Protocol):
    """Minimal interface every backend implements."""

    spec: object  # ModelSpec; typed loosely to avoid a circular import

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        prefill: str | None = None,
        n: int = 1,
    ) -> list[str]:
        """Return ``n`` assistant completions for ``messages``.

        If ``prefill`` is given, each completion is the continuation *after*
        ``prefill`` (the prefill itself is not included in the returned text),
        matching how the paper measures "generated continuation (excluding
        prefill)".
        """
        ...


class PrefillNotSupported(RuntimeError):
    """Raised when an API backend is asked to prefill an assistant turn."""
