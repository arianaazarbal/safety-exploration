"""Backend-agnostic model interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


Conversation = Sequence[ChatMessage]


class ModelClient(abc.ABC):
    """Common interface for chat generation.

    All emotion evaluations are run at ``temperature=1`` (paper default); the
    value is threaded through every call so configs can override it.
    """

    name: str

    @abc.abstractmethod
    def generate(
        self,
        conversation: Conversation,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        """Return the assistant's reply to ``conversation``."""

    def generate_batch(
        self,
        conversations: Sequence[Conversation],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> list[str]:
        """Default sequential batch; backends may override for true batching."""
        return [
            self.generate(c, temperature=temperature, max_tokens=max_tokens)
            for c in conversations
        ]

    def generate_with_prefill(
        self,
        conversation: Conversation,
        prefill: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        """Continue an assistant turn that begins with ``prefill``.

        Required for the Section 3 base-vs-instruct experiment. API backends
        that cannot truly prefill should raise :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support response prefilling."
        )

    @property
    def supports_prefill(self) -> bool:
        return False
