"""Model client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

from ..config import ModelSpec


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class ModelClient(ABC):
    """A chat-completion client for one model.

    ``chat`` returns ``n`` independent completions (assistant text). Generation
    is at the temperature/top_p the paper specifies (temperature 1 for the
    eval). ``continue_assistant`` is used by the Section 3 prefill experiment:
    it conditions on a partial final assistant turn and continues it, returning
    only the *generated* continuation (excluding the prefix).
    """

    spec: ModelSpec

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        n: int = 1,
    ) -> list[str]:
        ...

    def chat_batch(
        self,
        batch: list[list[Message]],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
    ) -> list[str]:
        """One completion per conversation in ``batch``.

        Default loops :meth:`chat`; backends that batch natively (vLLM) override
        this. Used by the rollout engine to advance many conversations through
        the same turn together.
        """
        return [
            self.chat(
                m, temperature=temperature, max_tokens=max_tokens, top_p=top_p, n=1
            )[0]
            for m in batch
        ]

    def continue_assistant(
        self,
        messages: list[Message],
        assistant_prefix: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        n: int = 1,
    ) -> list[str]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefill continuation"
        )
