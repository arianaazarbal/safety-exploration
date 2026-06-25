"""Backend-agnostic generation interface.

Every model in this replication is reachable through a :class:`ModelBackend`
that exposes two operations used by the rest of the code:

* :meth:`generate` — sample ``n`` completions for a chat ``messages`` list.
* :meth:`generate_with_prefill` — continue from a partial assistant turn
  (needed for the Section 3 prefill experiment and for base models, which are
  not chat-tuned and must be "primed" with a prefilled response).

Chat messages use the OpenAI-style ``{"role": ..., "content": ...}`` shape.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict

import config


class ChatMessage(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationConfig:
    temperature: float = config.TEMPERATURE
    top_p: float = config.TOP_P
    max_new_tokens: int = config.MAX_NEW_TOKENS
    # Disable hidden reasoning where the backend supports it (Appendix B.1:
    # "we set thinking to be false via the API").
    thinking: bool = False


class ModelBackend(ABC):
    """Uniform generation interface."""

    def __init__(self, spec):
        self.spec = spec

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        n: int = 1,
        cfg: GenerationConfig | None = None,
    ) -> list[str]:
        """Return ``n`` assistant completions for ``messages``."""

    @abstractmethod
    def generate_with_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        n: int = 1,
        cfg: GenerationConfig | None = None,
    ) -> list[str]:
        """Continue an assistant turn that already starts with ``prefill``.

        The returned strings are the *continuations only* (the prefill is
        stripped), matching the paper's "generated continuation (excluding
        prefill) is scored" protocol (Section 3.1).
        """

    # Convenience -----------------------------------------------------------
    @property
    def is_base(self) -> bool:
        return getattr(self.spec, "is_base", False)
