"""Abstract chat-model interface used by every experiment.

A ``Message`` is a ``{"role": ..., "content": ...}`` dict with role in
{"system", "user", "assistant"}. ``generate`` returns the assistant's text for the
next turn. ``prefill`` (optional) seeds the start of the assistant turn and the
model continues it -- this is what Section 3 / the recovery experiment rely on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import config


class Message(TypedDict):
    role: str
    content: str


class ChatModel(ABC):
    """Common interface for Gemma (local) and Gemini (OpenRouter) targets."""

    key: str
    family: str
    supports_prefill: bool = False

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefill: str | None = None,
    ) -> str:
        """Generate the next assistant turn.

        If ``prefill`` is given and the model supports it, the returned text is the
        continuation *excluding* the prefill (so callers can prepend it themselves
        when reconstructing the full turn, matching the paper's "score the generated
        continuation excluding prefill" rule).
        """
        raise NotImplementedError

    def generate_batch(
        self,
        batch: list[list[Message]],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefills: list[str | None] | None = None,
    ) -> list[str]:
        """Default sequential implementation; local backends may override for speed."""
        prefills = prefills or [None] * len(batch)
        return [
            self.generate(m, temperature=temperature, max_new_tokens=max_new_tokens,
                          prefill=p)
            for m, p in zip(batch, prefills)
        ]
