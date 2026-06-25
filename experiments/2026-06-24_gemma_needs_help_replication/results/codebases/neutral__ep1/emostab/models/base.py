"""Abstract chat-model interface shared by all backends."""
from __future__ import annotations

import abc
from typing import Dict, List

from ..config import ModelSpec

Message = Dict[str, str]          # {"role": ..., "content": ...}
Conversation = List[Message]


class ChatModel(abc.ABC):
    """Common interface for target models.

    Implementations must support:
      * batched multi-turn generation (`generate`)
      * optional assistant *prefill* -- continuing a partially written assistant
        turn -- which Section 3 relies on (only required for local models).
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.key = spec.key

    # --- single + batched generation -------------------------------------- #
    @abc.abstractmethod
    def generate_batch(
        self,
        conversations: List[Conversation],
        *,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        prefills: List[str] | None = None,
        seed: int | None = None,
    ) -> List[str]:
        """Generate one completion per conversation.

        If ``prefills`` is given (one string per conversation, possibly empty),
        the assistant turn is *seeded* with that text and the model continues it;
        the returned string is the continuation only (excluding the prefill).
        """

    def generate(
        self,
        conversation: Conversation,
        *,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        prefill: str | None = None,
        seed: int | None = None,
    ) -> str:
        prefills = [prefill or ""] if prefill is not None else None
        return self.generate_batch(
            [conversation],
            temperature=temperature,
            max_tokens=max_tokens,
            prefills=prefills,
            seed=seed,
        )[0]

    @property
    def supports_prefill(self) -> bool:
        return self.spec.supports_prefill

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.key} ({self.spec.model_id})>"
