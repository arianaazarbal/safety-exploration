"""Common chat-model interface.

A `Message` is a dict {"role": "user"|"assistant"|"system", "content": str}.
All target-model interaction in the experiments goes through this interface so
the rollout / prefill engines are provider-agnostic.
"""

from __future__ import annotations

import abc
from typing import TypedDict

from config import MAX_NEW_TOKENS, TEMPERATURE


class Message(TypedDict):
    role: str
    content: str


class ChatModel(abc.ABC):
    """Minimal chat interface used by all experiments."""

    name: str
    supports_prefill: bool = True

    @abc.abstractmethod
    def chat(self, messages: list[Message], *, n: int = 1,
             max_new_tokens: int = MAX_NEW_TOKENS,
             temperature: float = TEMPERATURE) -> list[str]:
        """Return `n` assistant completions for the conversation `messages`."""

    def prefill_continue(self, messages: list[Message], prefill: str, *,
                         n: int = 1, max_new_tokens: int = MAX_NEW_TOKENS,
                         temperature: float = TEMPERATURE) -> list[str]:
        """Continue an assistant turn that begins with `prefill`.

        Returns the continuation text only (the prefill is stripped). Used by the
        Section 3 base-vs-instruct prefilling experiment. Closed models that do
        not support assistant prefilling raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.name} does not support response prefilling")
