"""Model adapter interface.

Every provider integration implements `ModelAdapter.chat`, which takes a list of
role/content messages plus an optional system prompt and returns the assistant's
text. This is deliberately the smallest surface that the study needs — no tool
use, no streaming — so adding a provider is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ModelAdapter(Protocol):
    """Minimal chat interface implemented by every provider."""

    #: our study label for this model (e.g. "claude-opus")
    id: str
    #: the provider's model identifier (e.g. "claude-opus-4-8")
    model: str

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Return the assistant's text reply to `messages`."""
        ...


class AdapterError(RuntimeError):
    """Raised when a provider call fails in a way the runner should record."""
