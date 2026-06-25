"""Abstract chat-model interface shared by the local (Gemma) and API (Gemini)
backends.

The eval/rollout engine and the training/prefill pipelines all talk to models
through this interface, so swapping a backend never touches experiment code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class Message(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


class ChatModel(ABC):
    """A batched, multi-sample chat-completion model.

    Implementations must support sampling at ``temperature`` (the paper samples
    everything at temperature 1) and returning ``n`` independent completions per
    conversation in a single call so the 4000-response sweeps stay efficient.
    """

    name: str

    @abstractmethod
    def generate(
        self,
        conversations: list[list[Message]],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[list[str]]:
        """Append one assistant turn to each conversation.

        Returns, for each input conversation, a list of ``n`` completion strings.
        """

    def continue_assistant(
        self,
        conversations: list[list[Message]],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        n: int = 1,
    ) -> list[list[str]]:
        """Continue a *prefilled* final assistant turn (used by the Section 3
        prefill experiment).

        Each conversation's final message must have ``role == "assistant"``; the
        model continues from that text. The returned strings are the
        continuation only (excluding the prefill). Default raises so that API
        backends that cannot prefill fail loudly rather than silently diverging.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support assistant-prefill continuation"
        )

    def close(self) -> None:  # pragma: no cover - backends override if needed
        pass
