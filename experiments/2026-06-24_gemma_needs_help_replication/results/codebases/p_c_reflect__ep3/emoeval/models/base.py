"""Common chat-model interface shared by all backends.

A `Message` is a {"role": ..., "content": ...} dict where role is one of
"system" | "user" | "assistant". Backends translate this to their native
format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

Message = dict[str, str]


@dataclass
class GenerationResult:
    text: str
    raw: object | None = None          # backend-native response, for debugging


class ChatModel(Protocol):
    """Minimal interface every backend implements."""

    name: str

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        system: str | None = None,
    ) -> str:
        """Return the assistant's text reply to `messages`."""
        ...


class PrefillModel(Protocol):
    """Backends that can continue from an arbitrary assistant prefix.

    Required by the prefill experiments (Section 3) and recovery analysis
    (Section 4.2). Only the local HF backend implements this — base models
    have no chat API, and closed Gemini models cannot be prefilled.
    """

    name: str

    def continue_from(
        self,
        messages: Sequence[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a continuation of `prefill` given the conversation.

        Returns ONLY the newly generated text (excluding `prefill`), matching
        the paper's scoring of "the model-generated continuation, excluding the
        prefilled text".
        """
        ...
