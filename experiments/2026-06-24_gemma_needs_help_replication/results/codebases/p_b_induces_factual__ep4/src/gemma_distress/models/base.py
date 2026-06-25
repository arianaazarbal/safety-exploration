"""Model abstraction.

Two capabilities matter for this replication:

1. ``chat`` — multi-turn chat completion (Section 2 elicitation, all models).
2. ``continue_prefill`` — continue from a partial assistant turn (Section 3
   prefilling study, Section 4.2 recovery study). Only open-weight models
   (Gemma, and the Qwen/OLMo families the paper uses but we omit) support this;
   API-only models (Gemini) raise ``PrefillNotSupported``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class PrefillNotSupported(RuntimeError):
    """Raised when a prefill/continuation is requested from an API-only model."""


@runtime_checkable
class ChatModel(Protocol):
    """Common interface implemented by every backend in this package."""

    name: str          # canonical id, e.g. "gemma-3-27b-it"
    family: str        # "gemma" | "gemini"
    is_base_model: bool

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ) -> str:
        """Return the assistant's reply to the conversation."""
        ...

    def continue_prefill(
        self,
        messages: list[Message],
        prefill: str,
        *,
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ) -> list[str]:
        """Continue a partially-written assistant turn.

        ``prefill`` is text already "spoken" by the assistant; the model
        produces ``n`` continuations. The returned strings are continuations
        only (the prefill is *not* re-included), matching the paper's
        "generated continuation (excluding prefill) is scored" protocol.

        Backends that cannot prefill must raise ``PrefillNotSupported``.
        """
        ...
