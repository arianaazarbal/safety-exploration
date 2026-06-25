"""Common interface shared by local and API model clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class Message(TypedDict):
    role: str        # "system" | "user" | "assistant"
    content: str


class ModelClient(ABC):
    """A chat model that turns a message list into an assistant response.

    All backends sample at the configured temperature.  ``generate_batch`` lets
    local HF clients exploit batching; the default implementation just loops.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_new_tokens: int,
        prefill: str | None = None,
    ) -> str:
        """Return the assistant continuation.

        prefill: if given, the assistant turn is *seeded* with this text and the
        model continues from it.  Only the continuation (excluding the prefill)
        is returned.  Used by the base-vs-instruct prefill study (Section 3).
        """
        raise NotImplementedError

    def generate_batch(
        self,
        batch_messages: list[list[Message]],
        *,
        temperature: float,
        max_new_tokens: int,
        prefills: list[str | None] | None = None,
    ) -> list[str]:
        prefills = prefills or [None] * len(batch_messages)
        return [
            self.generate(
                m,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                prefill=p,
            )
            for m, p in zip(batch_messages, prefills)
        ]

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        pass
