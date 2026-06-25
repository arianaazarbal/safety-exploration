"""Abstract model-client interface shared by all backends.

The harness only needs two capabilities from a subject model:

* ``chat`` -- given a list of messages, produce the next assistant message.
* ``continue_from_prefill`` -- given messages *and* a partial assistant turn,
  continue generating from that prefill (needed for Section 3 and the recovery
  experiment). Only open / local models support this.

Judges add a ``complete`` method (single prompt -> text); see
:class:`~emotional_instability.models.anthropic_judge.AnthropicClient`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    role: str           # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    # Optional token-level detail (used by truncation experiments). ``token_ids``
    # are the *new* tokens only; ``token_texts`` are their decoded pieces.
    token_ids: Optional[list[int]] = None
    token_texts: Optional[list[str]] = None
    finish_reason: Optional[str] = None
    meta: dict = field(default_factory=dict)


class ModelClient(abc.ABC):
    """A subject model that can carry on a multi-turn chat."""

    supports_prefill: bool = False

    @abc.abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        """Generate the next assistant turn given the conversation so far."""

    def continue_from_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        """Continue generating, treating ``prefill`` as the start of the assistant
        turn. Returns only the *continuation* (excluding the prefill), matching
        the paper's "generated continuation (excluding prefill) is scored"
        protocol (Sec 3.1)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefill-based continuation."
        )

    def tokenize(self, text: str) -> list[int]:
        raise NotImplementedError

    def detokenize(self, token_ids: list[int]) -> str:
        raise NotImplementedError
