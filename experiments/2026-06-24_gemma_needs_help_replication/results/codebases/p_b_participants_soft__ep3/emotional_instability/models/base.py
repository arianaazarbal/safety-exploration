"""Common chat-model interface used across all experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


Conversation = Sequence[Message]


class ChatModel(ABC):
    """Backend-agnostic chat model.

    `temperature` defaults to the paper's value (1.0) at call sites; we do not
    bake it in here so capability evals can override it.
    """

    key: str

    @abstractmethod
    def generate(
        self,
        messages: Conversation,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Return a single completion for the final (assistant) turn."""

    def prefill_continue(
        self,
        messages: Conversation,
        prefill: str,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        """Continue an assistant turn that begins with `prefill`.

        Returns ONLY the generated continuation (excluding the prefill text),
        matching the paper's Section 3.1 measurement ("The generated
        continuation (excluding prefill) is scored by the judge").

        Not all backends support prefilling (closed API models often don't);
        those raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefill continuation"
        )

    def batch_generate(
        self,
        conversations: Sequence[Conversation],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        stop: Optional[list[str]] = None,
    ) -> list[str]:
        """Default: sequential. HF/API clients override with real batching."""
        return [
            self.generate(c, temperature, max_new_tokens, stop)
            for c in conversations
        ]
