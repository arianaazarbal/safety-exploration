"""Model client interfaces.

The pipeline talks to models through small capability interfaces so that the
same orchestration code works for local Gemma (HuggingFace/vLLM) and API
models (Gemini, Claude, GPT). Not every provider supports every capability:

  * :class:`ChatModel`     - multi-turn chat completion (all providers).
  * :class:`PrefillModel`  - continue a partially written assistant turn
                             (needed for Section 3 prefilling; local only).
  * :class:`ResidualModel` - expose per-layer residual-stream activations
                             (needed for Appendix I; local transformers only).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    import numpy as np

    from gemma_distress.conversations import Message


class ChatModel(ABC):
    """Multi-turn chat completion."""

    name: str

    @abstractmethod
    def chat(
        self,
        messages: Sequence["Message"],
        temperature: float = 1.0,
        max_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        """Return the assistant completion for ``messages``."""

    def chat_batch(
        self,
        batch: Sequence[Sequence["Message"]],
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> list[str]:
        """Batched chat. Default falls back to sequential ``chat`` calls."""
        return [
            self.chat(messages, temperature=temperature, max_tokens=max_tokens)
            for messages in batch
        ]


class PrefillModel(Protocol):
    """Continue a pre-written assistant turn (assistant-message prefill)."""

    name: str

    def continue_assistant(
        self,
        messages: Sequence["Message"],
        prefill: str,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        """Return ONLY the generated continuation (excluding ``prefill``).

        ``messages`` ends with the user turn; ``prefill`` is the start of the
        assistant turn that the model must continue from. Base (pretrained)
        models continue naturally; instruct models continue their own turn.
        """
        ...

    def continue_assistant_batch(
        self,
        messages: Sequence["Message"],
        prefill: str,
        n: int,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> list[str]:
        ...


class ResidualModel(Protocol):
    """Expose residual-stream activations for logit-lens emotion detection."""

    name: str

    def residual_stream(
        self, text: str
    ) -> "np.ndarray":  # shape (n_layers, n_tokens, d_model)
        ...

    def unembed(self, residual: "np.ndarray") -> "np.ndarray":
        """Project residual-stream vectors to vocab logits."""
        ...

    def tokenize(self, text: str) -> list[int]:
        ...
