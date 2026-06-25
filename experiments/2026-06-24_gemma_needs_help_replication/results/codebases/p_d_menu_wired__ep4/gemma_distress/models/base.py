"""Abstract subject-model interface.

A *subject model* is one of the models under evaluation (Gemma or Gemini).
Concrete implementations live in :mod:`gemma_distress.models.gemma` and
:mod:`gemma_distress.models.gemini`.

The interface is deliberately small:

* ``generate`` — standard chat completion from a message list.
* ``generate_with_prefill`` — continue from a partially-written assistant turn,
  used by the §3 base-vs-instruct prefilling experiment. Base models are not
  chat-tuned, so prefilling is how we get them to "continue" comparably.

Messages are plain dicts ``{"role": "user"|"assistant"|"system", "content": str}``
to stay uniform across backends.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable

from ..config import SamplingConfig

Message = dict  # {"role": str, "content": str}


@dataclass
class GenerationResult:
    """Result of a single generation call."""

    text: str
    # Token ids of the generated continuation, when the backend exposes them
    # (HF does; Gemini does not). Used by onset-truncation in §3.
    token_ids: list[int] | None = None


class SubjectModel(abc.ABC):
    """Common interface for every evaluated model."""

    #: Short, stable identifier used in output filenames and metrics tables.
    name: str

    #: Whether the backend supports native function/tool calling. When True the
    #: welfare opt-out can be exposed as a tool; otherwise it falls back to the
    #: sentinel-string mechanism.
    supports_tools: bool = False

    @abc.abstractmethod
    def generate(self, messages: list[Message], cfg: SamplingConfig) -> GenerationResult:
        """Generate an assistant turn given a message history."""

    def generate_with_prefill(
        self, messages: list[Message], prefill: str, cfg: SamplingConfig
    ) -> GenerationResult:
        """Continue an assistant turn that has been pre-filled with ``prefill``.

        Returns *only the continuation* (excluding the prefill), matching the
        paper's scoring of "the generated continuation (excluding prefill)".
        Backends that cannot prefill should raise ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled generation"
        )

    def tokenize(self, text: str) -> list[int]:
        """Tokenise text (for onset/early truncation). Optional per backend."""
        raise NotImplementedError

    def detokenize(self, token_ids: Iterable[int]) -> str:
        """Inverse of :meth:`tokenize`. Optional per backend."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - cleanup hook
        """Release any held resources (GPU memory, sessions)."""
