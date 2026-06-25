"""Abstract chat-model interface shared by all participant backends.

A conversation is a list of ``Message`` dicts with roles ``system`` / ``user`` /
``assistant`` (the assistant turns are the model's own prior responses in a
multi-turn rollout). Backends must implement two operations:

* :meth:`generate` — produce the next assistant turn given the conversation.
* :meth:`continue_from_prefill` — given a conversation plus a partial assistant
  turn (the *prefill*), continue generating *only the continuation*. This is the
  primitive the Section 3 base-vs-instruct prefill experiment relies on; base
  models that were never chat-tuned still continue text reliably from a prefill.
"""

from __future__ import annotations

import abc
from typing import TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatModel(abc.ABC):
    """Uniform generation interface over a participant model."""

    name: str

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[str]:
        """Return ``n`` sampled assistant completions for ``messages``.

        Implementations should return exactly ``n`` strings (the raw assistant
        text, with any chat-template scaffolding stripped).
        """

    def generate_one(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        return self.generate(
            messages, temperature=temperature, max_new_tokens=max_new_tokens, n=1
        )[0]

    def continue_from_prefill(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[str]:
        """Continue from a partial assistant turn (the *prefill*).

        Returns ``n`` continuations *excluding* the prefill text. Only local
        backends (HF / LoRA) support genuine prefill continuation; API backends
        without an explicit assistant-prefill channel raise
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefill continuation"
        )

    @property
    def supports_prefill(self) -> bool:
        return False
