"""Model backend interface.

All target models, judges, and auditors are accessed through :class:`ChatModel`, which
hides whether a model runs locally (HuggingFace / vLLM) or behind an API (OpenRouter,
Anthropic). The interface is deliberately small:

* :meth:`chat_batch`        — sample completions for a batch of conversations.
* :meth:`continue_from_prefill` — continue an assistant turn from a fixed prefix. Used by
                              the Section 3 prefill experiment; only the local backends
                              support it (base models have no chat template, and closed
                              API models do not expose true assistant prefilling).

A *conversation* is a list of ``{"role": ..., "content": ...}`` messages with roles
``system`` / ``user`` / ``assistant``, matching the Anthropic and OpenAI message shapes
and the Gemma chat template.
"""

from __future__ import annotations

import abc
from typing import Optional

Message = dict[str, str]
Conversation = list[Message]


class ChatModel(abc.ABC):
    """Abstract chat model. Subclasses implement batched generation."""

    #: Whether :meth:`continue_from_prefill` is supported by this backend.
    supports_prefill: bool = False

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def chat_batch(
        self,
        conversations: list[Conversation],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
    ) -> list[list[str]]:
        """Return ``n`` completions for each conversation.

        The outer list is per-conversation, the inner list is the ``n`` samples for that
        conversation. Implementations should preserve order.
        """

    def chat(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Sample a single completion for one conversation."""
        return self.chat_batch(
            [conversation], temperature=temperature, max_new_tokens=max_new_tokens, n=1
        )[0][0]

    def chat_n(
        self,
        conversation: Conversation,
        *,
        n: int,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]:
        """Sample ``n`` completions for one conversation."""
        return self.chat_batch(
            [conversation], temperature=temperature, max_new_tokens=max_new_tokens, n=n
        )[0]

    def continue_from_prefill(
        self,
        conversation: Conversation,
        prefill: str,
        *,
        n: int,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]:
        """Continue the final assistant turn from ``prefill`` (excluded from the return).

        Returns the ``n`` generated continuations *without* the prefill prepended, matching
        the paper's protocol of scoring only the model-generated portion.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilling."
        )

    def close(self) -> None:  # pragma: no cover - resource cleanup hook
        """Release backend resources (GPU memory, HTTP sessions)."""
