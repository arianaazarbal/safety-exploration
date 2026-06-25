"""Common chat-model interface.

Every participant model (Gemma via HF, Gemini via OpenRouter) implements
``ChatModel`` so the rollout engine is backend-agnostic. ``Message`` is the
minimal OpenAI/Anthropic-style dict: ``{"role": ..., "content": ...}``.
"""
from __future__ import annotations

import abc
from typing import Optional, TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatModel(abc.ABC):
    """A multi-turn chat model that produces one assistant turn at a time."""

    name: str

    @abc.abstractmethod
    def chat(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
    ) -> str:
        """Return a single assistant completion for ``messages``.

        If ``prefill`` is given, the assistant turn is *seeded* with that text and
        the model continues from it; the returned string EXCLUDES the prefill
        (callers that want the full turn concatenate it themselves). Prefill is
        required for the Section 3 base-vs-instruct experiment and is only
        supported by local backends.
        """
        raise NotImplementedError

    def supports_prefill(self) -> bool:
        return False
