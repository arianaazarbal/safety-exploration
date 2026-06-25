"""Abstract chat-client interface shared by all target backends.

A ``Message`` is a plain ``{"role": "user"|"assistant"|"system", "content": str}``
dict, matching both the Anthropic/OpenAI message shape and the HF chat-template
shape. The rollout engine speaks only in these dicts, so the same multi-turn
loop drives Gemma (local) and Gemini (API) without branching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class Message(TypedDict):
    role: str        # "system" | "user" | "assistant"
    content: str


class ChatClient(ABC):
    """Common interface for sampling assistant turns from a target model."""

    #: Stable identifier recorded alongside every response for provenance.
    name: str

    @abstractmethod
    def chat(self, messages: list[Message], *, temperature: float, max_new_tokens: int) -> str:
        """Sample a single assistant turn given the conversation so far.

        Returns the assistant's text only (no role markers / template tokens).
        """

    # ---- Optional capabilities (prefilling, tokenisation) ------------------
    # Only the local Gemma backend implements these; API backends raise
    # NotImplementedError. Section 3 prefilling is Gemma-only by design.

    def continue_text(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Continue an assistant turn that has been prefilled with ``prefill``.

        Returns only the *generated continuation* (excluding the prefill), per
        the Section 3.1 protocol ("the generated continuation (excluding
        prefill) is scored by the judge").
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled continuation."
        )

    def count_tokens(self, text: str) -> int:
        """Number of tokens ``text`` encodes to under this model's tokenizer."""
        raise NotImplementedError

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        """Return the prefix of ``text`` that is the first ``n_tokens`` tokens."""
        raise NotImplementedError
