"""Common chat-model interface shared by local and API target models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class Generation:
    """Result of a single model turn."""

    text: str
    # When prefilling a base model, `text` is the *continuation only* (the
    # prefill is stored separately so the judge scores only generated tokens).
    prefill: Optional[str] = None
    finish_reason: Optional[str] = None


class ChatModel(Protocol):
    """Minimal interface every target model implements."""

    name: str

    def generate(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> Generation:
        """Generate one assistant turn given the conversation so far."""
        ...

    def generate_with_prefill(
        self,
        messages: Sequence[ChatMessage],
        prefill: str,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> Generation:
        """Generate a continuation of `prefill` (Section 3). API models that do
        not support prefilling should raise NotImplementedError."""
        ...
