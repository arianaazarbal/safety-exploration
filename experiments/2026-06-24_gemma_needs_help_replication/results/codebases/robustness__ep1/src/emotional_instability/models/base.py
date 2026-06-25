"""Backend abstraction shared by local (HF) and API (Gemini) target models.

A ``Message`` is ``{"role": "user"|"assistant", "content": str}``. System content
is passed separately because Gemma's chat template has no system role and we want
identical handling across backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

Message = dict  # {"role": str, "content": str}


class ModelBackend(ABC):
    """Generates assistant turns given a conversation."""

    name: str

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: Optional[int] = None,
    ) -> str:
        """Return a single assistant completion for the conversation."""

    # -- prefill support (Section 3). Only required for local models. ----------
    def continue_from(
        self,
        messages: list[Message],
        prefill: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: Optional[int] = None,
    ) -> str:
        """Continue an assistant turn that has been prefilled with ``prefill``.

        Returns ONLY the newly generated continuation (excluding the prefill),
        matching the paper's scoring of "the generated continuation (excluding
        prefill)" (Section 3.1).
        """
        raise NotImplementedError(f"{self.name} does not support prefill continuation")

    def count_tokens(self, text: str) -> int:
        """Token count under this model's tokenizer (for early-truncation)."""
        raise NotImplementedError

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        """Return the first ``n_tokens`` tokens of ``text`` decoded back to a string."""
        raise NotImplementedError
