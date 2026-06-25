"""The chat-model abstraction shared by every backend.

A ``Message`` is the familiar ``{"role": ..., "content": ...}`` dict. Roles are
``"system"``, ``"user"``, ``"assistant"``. Every backend implements
``generate`` (one completion) and inherits ``generate_batch``.

Two capabilities beyond plain chat are needed by the experiments:

* **Assistant prefill** — Section 3 continues a partially-written assistant
  turn. ``generate(prefill=...)`` appends ``prefill`` to the prompt and asks the
  model to continue it; the returned text excludes the prefill.
* **Tokenisation** — the prefill truncation points are defined in *tokens*
  ("20 tokens into the turn"). Backends that own a tokenizer expose
  ``count_tokens`` / ``decode_first_tokens``; API backends raise
  ``NotImplementedError`` (the prefill experiment only runs on local Gemma).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterable, Sequence

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass
class GenerationResult:
    """One model completion plus light metadata."""

    text: str
    prompt_messages: list[Message] = field(default_factory=list)
    prefill: str | None = None
    finish_reason: str | None = None
    raw: object | None = None  # provider response object, for debugging


class ChatModel(abc.ABC):
    """Backend-agnostic chat model."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self.key = spec.key
        self.family = spec.family

    # -- generation -------------------------------------------------------- #
    @abc.abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        """Return a single completion for ``messages``.

        If ``prefill`` is given, the assistant response is seeded with that text
        and the model continues it; the returned ``text`` is the *continuation
        only* (prefill stripped), matching the paper's "generated continuation
        (excluding prefill) is scored by the judge" (Section 3.1).
        """

    def generate_batch(
        self,
        batch: Sequence[Sequence[Message]],
        *,
        temperature: float,
        max_tokens: int,
        prefills: Sequence[str | None] | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[GenerationResult]:
        """Default sequential implementation; backends may override for speed."""
        prefills = prefills or [None] * len(batch)
        return [
            self.generate(
                msgs, temperature=temperature, max_tokens=max_tokens,
                prefill=pf, stop=stop,
            )
            for msgs, pf in zip(batch, prefills)
        ]

    # -- tokenisation (local backends only) -------------------------------- #
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} has no local tokenizer; token-level "
            "truncation is only supported for HF backends.")

    def decode_first_tokens(self, text: str, n: int) -> str:
        """Return the prefix of ``text`` consisting of its first ``n`` tokens."""
        raise NotImplementedError(
            f"{type(self).__name__} has no local tokenizer.")

    # -- lifecycle --------------------------------------------------------- #
    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (GPU memory, HTTP sessions). Optional."""
