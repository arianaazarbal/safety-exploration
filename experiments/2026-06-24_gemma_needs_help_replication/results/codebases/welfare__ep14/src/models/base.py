"""Backend-agnostic chat model abstraction.

Every experiment in this repo talks to models through :class:`ChatModel`. The
two capabilities the paper needs that go beyond "send messages, get text" are:

1. **Prefilling** (Section 3 / recovery test): force the model to *continue* a
   given assistant prefix rather than start a fresh turn. ``generate`` exposes
   this via the ``prefill`` argument.
2. **Batched sampling at T=1**: ``generate_batch`` lets a backend parallelise
   (HF batch / async API) while callers stay simple.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Sequence, TypedDict

import config


class Message(TypedDict):
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationParams:
    temperature: float = config.TEMPERATURE
    top_p: float = config.TOP_P
    max_new_tokens: int = config.MAX_NEW_TOKENS
    stop: Sequence[str] = field(default_factory=tuple)
    seed: int | None = None
    # Contract is one completion per input conversation. To draw N samples of a
    # prompt, callers replicate the conversation N times in the batch.


class ChatModel(abc.ABC):
    """A chat model that can complete conversations, optionally with a prefill."""

    def __init__(self, spec: "config.ModelSpec"):
        self.spec = spec
        self.key = spec.key

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        params: GenerationParams | None = None,
        prefill: str | None = None,
    ) -> str:
        """Return a single completion.

        If ``prefill`` is given, the returned string is the *continuation only*
        (the prefill is not echoed back), matching how Section 3 scores
        "the model-generated continuation, excluding the prefilled text".
        """

    def generate_batch(
        self,
        batch: list[list[Message]],
        params: GenerationParams | None = None,
        prefills: list[str | None] | None = None,
    ) -> list[str]:
        """Default sequential implementation; backends override for throughput."""
        prefills = prefills or [None] * len(batch)
        return [
            self.generate(msgs, params, prefill=pf)
            for msgs, pf in zip(batch, prefills)
        ]

    def close(self) -> None:  # pragma: no cover - backends override if needed
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
