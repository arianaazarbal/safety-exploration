"""Abstract chat-model interface shared by all backends.

Two capabilities matter for this paper:

1. ``generate`` — standard multi-turn chat sampling (the elicitation loop).
2. ``continue_prefill`` — append/continue from a partially-written assistant
   turn (the Section 3 base-vs-instruct prefill experiment). Base models have no
   chat template, so this is the only fair way to compare them with instruct
   models.

Not every backend supports prefill (API providers generally do not expose raw
continuation), so ``continue_prefill`` raises ``NotImplementedError`` where
unsupported and the prefill experiment selects local backends accordingly.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, TypedDict


class ChatMessage(TypedDict):
    role: str       # "system" | "user" | "assistant"
    content: str


@dataclass
class Completion:
    text: str
    # Optional metadata for downstream analysis / debugging.
    finish_reason: Optional[str] = None
    raw: Optional[dict] = None


class ChatModel(abc.ABC):
    """A sampler over chat conversations."""

    def __init__(self, spec):
        self.spec = spec

    @abc.abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> list[Completion]:
        """Sample `n` completions for the final assistant turn."""

    def continue_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> list[Completion]:
        """Continue an assistant turn that already starts with `prefill`.

        Returns ONLY the newly generated continuation (excluding the prefill),
        matching the paper's "generated continuation (excluding prefill) is
        scored by the judge".
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefill continuation."
        )

    @property
    def supports_prefill(self) -> bool:
        return type(self).continue_prefill is not ChatModel.continue_prefill

    def close(self):  # pragma: no cover - backend specific
        pass
