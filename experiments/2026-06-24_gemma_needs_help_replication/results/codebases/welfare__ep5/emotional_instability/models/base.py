"""Provider-agnostic chat interface.

A ``Message`` is a ``{"role": ..., "content": ...}`` dict with role in
{"system", "user", "assistant"}. All clients implement :meth:`generate` for a
full assistant turn and, where possible, :meth:`continue_prefill` for the
Section 3 prefilling experiment (continue a partially-written assistant turn).
"""

from __future__ import annotations

import abc
from typing import Optional, Sequence, TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatModel(abc.ABC):
    """Abstract chat model."""

    #: short display name (matches config.ModelSpec.name)
    name: str

    #: whether continue_prefill is meaningful for this backend
    supports_prefill: bool = False

    @abc.abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[str]:
        """Generate ``n`` assistant continuations of ``messages``.

        Returns a list of ``n`` strings (the assistant turn text only).
        """

    def continue_prefill(
        self,
        messages: Sequence[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        n: int = 1,
    ) -> list[str]:
        """Continue a partially written final assistant turn.

        ``prefill`` is text already "spoken" by the assistant; the returned
        strings are *only the continuation* (excluding the prefill), matching
        the paper's scoring of "the generated continuation (excluding prefill)".

        Backends that cannot prefill (e.g. closed Gemini) raise
        ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilling (closed-weight API)."
        )

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        pass


def prefill_supported(model: ChatModel) -> bool:
    return bool(getattr(model, "supports_prefill", False))
