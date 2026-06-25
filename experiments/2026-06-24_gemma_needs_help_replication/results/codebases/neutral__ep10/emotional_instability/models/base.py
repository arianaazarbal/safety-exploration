"""Abstract chat-model interface shared by the HF and API backends.

The elicitation eval only needs `chat`. The prefill experiment (Section 3)
additionally needs `continue_prefill` (continue an assistant turn from a fixed
prefix), which only the local HF backend can support. The internal-emotion
probe (Appendix I) needs `residual_logits`, again HF-only. API backends raise
NotImplementedError for those.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:
    role: str        # "system" | "user" | "assistant"
    content: str


class ChatModel(abc.ABC):
    """Minimal contract every backend implements."""

    def __init__(self, spec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    # --- core generation -------------------------------------------------- #
    @abc.abstractmethod
    def chat(
        self,
        messages: list[Message],
        max_new_tokens: int,
        temperature: float,
        seed: Optional[int] = None,
    ) -> str:
        """Return the assistant's reply to a conversation."""

    def chat_batch(
        self,
        conversations: list[list[Message]],
        max_new_tokens: int,
        temperature: float,
        seeds: Optional[list[int]] = None,
    ) -> list[str]:
        """Default: sequential. HF/vLLM backends override with real batching."""
        seeds = seeds or [None] * len(conversations)
        return [
            self.chat(c, max_new_tokens, temperature, seed=s)
            for c, s in zip(conversations, seeds)
        ]

    # --- prefill (Section 3) --------------------------------------------- #
    def continue_prefill(
        self,
        messages: list[Message],
        prefill: str,
        max_new_tokens: int,
        temperature: float,
        seed: Optional[int] = None,
    ) -> str:
        """Continue the final assistant turn from `prefill`.

        Returns only the *generated continuation* (excluding the prefill), per
        the paper's scoring convention. Base models (no chat template) use the
        same hook with raw concatenation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilling."
        )

    # --- internal probe (Appendix I) ------------------------------------- #
    def residual_logits(self, text: str, layers: list[int]):
        """Unembed the residual stream at `layers` for every token in `text`.

        Returns (token_ids, {layer: logits[seq, vocab]}). HF-only.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose internal activations."
        )

    def close(self) -> None:  # pragma: no cover - cleanup hook
        pass
