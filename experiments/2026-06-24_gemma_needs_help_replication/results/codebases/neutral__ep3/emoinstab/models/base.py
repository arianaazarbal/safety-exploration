"""Abstract model-client interface.

Every backend (vLLM, transformers, OpenRouter, Anthropic) implements this
interface so the experiment code is backend-agnostic. The two capabilities the
experiments need beyond plain chat are:

  * batched generation (``chat_batch``) -- thousands of rollouts.
  * prefill continuation (``continue_prefill``) -- Section 3 needs the model to
    *continue* a partially-written assistant turn rather than start a new one.
    For base (pretrained) models this is the only sensible way to interact.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from ..config import GenConfig, DEFAULT_GEN
from ..data_types import Conversation


@dataclass
class GenResult:
    text: str
    raw: Optional[dict] = None     # backend-specific metadata (token ids, usage)


class ModelClient(ABC):
    """Interface implemented by all backends."""

    name: str
    supports_prefill: bool = False

    # --- chat ----------------------------------------------------------- #
    @abstractmethod
    def chat(self, messages: Conversation, gen: GenConfig = DEFAULT_GEN) -> GenResult:
        """Generate one assistant response for a conversation."""

    def chat_batch(
        self,
        batch: Sequence[Conversation],
        gen: GenConfig = DEFAULT_GEN,
    ) -> list[GenResult]:
        """Generate responses for a batch of conversations.

        The default implementation loops; backends with native batching
        (vLLM) override this for throughput.
        """
        return [self.chat(m, gen) for m in batch]

    # --- prefill continuation ------------------------------------------- #
    def continue_prefill(
        self,
        messages: Conversation,
        prefill: str,
        gen: GenConfig = DEFAULT_GEN,
    ) -> GenResult:
        """Continue an assistant turn that begins with ``prefill``.

        The returned text is the *continuation only* (excluding ``prefill``),
        matching the paper's scoring convention (Section 3.1).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefill continuation."
        )

    def continue_prefill_batch(
        self,
        batch: Sequence[tuple[Conversation, str]],
        gen: GenConfig = DEFAULT_GEN,
    ) -> list[GenResult]:
        return [self.continue_prefill(m, p, gen) for (m, p) in batch]

    # --- tokenisation (needed for "early"/"onset" truncation) ----------- #
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        """Return the prefix of ``text`` consisting of its first ``n_tokens``."""
        raise NotImplementedError
