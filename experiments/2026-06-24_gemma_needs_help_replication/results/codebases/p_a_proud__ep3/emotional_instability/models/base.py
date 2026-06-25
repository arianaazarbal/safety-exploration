"""Model client abstraction.

Every backend (local Gemma, Gemini-via-OpenRouter, Claude-via-Anthropic) exposes
the same small surface:

* :meth:`chat` — multi-turn chat completion for instruct / API models.
* :meth:`complete` — raw text continuation from a prefix, used by the §3 prefill
  methodology (base models are not chat-tuned, so we prefill and continue).

The distress evaluations only need :meth:`chat`; the prefill experiments need
:meth:`complete`; the internal-emotion probing (Appendix I) needs the underlying
HF model and tokenizer, which only the local backend exposes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Sequence, TypedDict

from ..config import SamplingConfig


class ChatMessage(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    # Optional usage / metadata for cost tracking and debugging.
    finish_reason: str | None = None
    raw: dict | None = None


class ModelClient(abc.ABC):
    """Common interface for all model backends."""

    name: str

    @abc.abstractmethod
    def chat(
        self, messages: Sequence[ChatMessage], sampling: SamplingConfig
    ) -> GenerationResult:
        """Generate the assistant's reply to a chat-formatted conversation."""

    def chat_batch(
        self, conversations: Sequence[Sequence[ChatMessage]], sampling: SamplingConfig
    ) -> list[GenerationResult]:
        """Default batched chat: sequential. Local backends override for speed."""
        return [self.chat(conv, sampling) for conv in conversations]

    def complete(self, prefix: str, sampling: SamplingConfig) -> GenerationResult:
        """Continue raw ``prefix`` text. Required for the prefill experiment.

        API chat backends generally cannot do true text continuation; they raise
        :class:`NotImplementedError` here. The §3 base-vs-instruct comparison is
        therefore local-only (Gemma), which matches the paper's scope (Gemini has
        no public base model anyway).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support raw text completion; "
            "the prefill experiment requires a local backend."
        )

    def supports_completion(self) -> bool:
        return type(self).complete is not ModelClient.complete

    def count_tokens(self, text: str) -> int:
        """Token count using the model's own tokenizer where available.

        Falls back to a whitespace-word count, which is adequate for the
        coarse truncation points the prefill experiment uses (20 tokens / onset).
        """
        return len(text.split())
