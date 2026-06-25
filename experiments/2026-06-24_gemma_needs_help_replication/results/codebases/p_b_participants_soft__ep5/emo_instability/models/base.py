"""Shared chat-client interface.

A ``Message`` is a simple ``{"role", "content"}`` dict (roles: ``system`` /
``user`` / ``assistant``). All clients accept a list of messages and return the
assistant's completion text.

Two capabilities matter for this replication:

* ``generate`` — standard multi-turn generation (Section 2 rollouts).
* ``prefill_continue`` — continue from a partially-written assistant message
  (Section 3 base-vs-instruct experiment). Only local models support this.
"""
from __future__ import annotations

from typing import Any, TypedDict


class Message(TypedDict):
    role: str           # "system" | "user" | "assistant"
    content: str


class ChatClient:
    """Abstract chat client. Subclasses implement ``generate`` (and optionally
    ``prefill_continue``)."""

    #: human-readable model name, used in result manifests
    name: str = "abstract"
    #: whether this client can continue from a prefilled assistant turn
    supports_prefill: bool = False

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    def prefill_continue(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """Continue an assistant turn that already begins with ``prefill``.

        Returns ONLY the newly generated continuation (excluding ``prefill``),
        matching the paper's scoring of "the generated continuation (excluding
        prefill)". Default raises; local clients override.
        """
        raise NotImplementedError(
            f"{self.name} ({type(self).__name__}) does not support prefilling."
        )

    def close(self) -> None:  # pragma: no cover - resource cleanup hook
        pass
