"""Shared chat-model abstraction.

A `Message` is a single chat turn. A `ChatModel` takes a conversation (a list of
messages) and returns the next assistant message's text. This is all the
elicitation harness needs: it builds up the conversation turn by turn, calling
`generate` after each user rejection.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ChatModel(abc.ABC):
    """Interface every target/judge model client implements."""

    name: str

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return the assistant's reply to `messages`. Single completion."""

    def generate_batch(
        self,
        conversations: list[list[Message]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """Default: sequential. Local backends override this for true batching."""
        return [
            self.generate(c, temperature=temperature, max_tokens=max_tokens)
            for c in conversations
        ]
