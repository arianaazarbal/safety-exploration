"""The ``ChatModel`` interface that every backend implements.

Two generation modes matter for this paper:

* ``chat`` — standard multi-turn generation (Section 2 elicitation, judges).
* ``prefill`` — the model's assistant turn is *seeded* with given text and it
  continues from there (Section 3 base-vs-instruct comparison, Section 4.2
  recovery). For instruct models this seeds the final assistant turn; for base
  models, which have no chat template, the whole conversation is rendered as
  plain text and continued.

Backends that cannot support prefill (remote APIs like Gemini, which do not
expose assistant-turn continuation) raise ``NotImplementedError`` — the prefill
experiments are therefore Gemma-only, exactly as in the paper (Gemini has no
public base model and no continuation API).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..types import Message


class GenerationError(RuntimeError):
    """Raised when a backend permanently fails to produce a completion."""


class ChatModel(ABC):
    """Common interface for target models, judges, and auditors."""

    name: str
    supports_prefill: bool = False

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        """Return the assistant's reply to ``messages`` (system/user/assistant)."""

    def generate_with_prefill(
        self,
        messages: Sequence[Message],
        prefill: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Continue an assistant turn seeded with ``prefill``.

        Returns only the *generated continuation*, excluding the prefill text
        (Paper §3.1: "The model-generated continuation, excluding the prefilled
        text, is scored").
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled generation."
        )

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        """Release any held resources (GPU memory, HTTP sessions)."""
