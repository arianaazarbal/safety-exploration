"""Abstract target-model interface.

The elicitation protocol (Section 2) and the prefilling protocol (Section 3)
need three capabilities from a target model:

1. ``chat`` -- generate the next assistant turn given a conversation. Used for
   every multi-turn rejection rollout.
2. ``continue_from`` -- continue a *prefilled* assistant turn. Used by the
   base-vs-instruct prefilling experiment, and also lets base models (which were
   never chat-tuned) produce on-distribution continuations.
3. ``count_tokens`` / ``truncate_to_tokens`` -- needed to truncate seed
   responses at "20 tokens in" and at the labelled emotional onset (Section 3.1).

Gemini, being a closed API, cannot truly prefill an assistant turn or expose a
tokenizer; ``GeminiClient`` raises ``PrefillUnsupported`` for (2)/(3). This is
why Section 3 is Gemma-only in our scope (see DESIGN.md).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypedDict


class Turn(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


class PrefillUnsupported(NotImplementedError):
    """Raised by backends that cannot prefill an assistant turn (e.g. Gemini)."""


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ModelClient(ABC):
    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id

    @abstractmethod
    def chat(
        self,
        messages: list[Turn],
        *,
        temperature: float,
        max_new_tokens: int,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate the next assistant turn for ``messages``."""

    def continue_from(
        self,
        messages: list[Turn],
        prefix: str,
        *,
        temperature: float,
        max_new_tokens: int,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> GenerationResult:
        """Continue an assistant turn that already begins with ``prefix``.

        Returns only the *newly generated* text (excluding ``prefix``), matching
        the paper's scoring of "the generated continuation (excluding prefill)".
        Backends that cannot prefill raise ``PrefillUnsupported``.
        """
        raise PrefillUnsupported(f"{type(self).__name__} cannot prefill assistant turns")

    def count_tokens(self, text: str) -> int:
        raise PrefillUnsupported(f"{type(self).__name__} does not expose a tokenizer")

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        raise PrefillUnsupported(f"{type(self).__name__} does not expose a tokenizer")

    def close(self) -> None:  # optional resource cleanup (frees GPU memory)
        pass
