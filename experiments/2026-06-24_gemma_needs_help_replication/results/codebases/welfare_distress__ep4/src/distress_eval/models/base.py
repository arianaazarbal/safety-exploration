"""Common chat-model interface used by both target models and judges."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatModel(ABC):
    """Minimal multi-turn chat interface.

    Implementations must be safe to call from multiple threads (the runner uses
    a thread pool). Clients that are not thread-safe should construct per-call
    resources or hold a lock internally.
    """

    def __init__(self, key: str, model: str):
        self.key = key
        self.model = model

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return the assistant's text completion for `messages`."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} key={self.key!r} model={self.model!r}>"
