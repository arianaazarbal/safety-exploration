"""Backend-agnostic chat interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict

from ..config import ModelSpec, SamplingConfig


class Message(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    """A single generated continuation.

    `text` excludes any prefill that was supplied (the rollout/prefill engines
    only ever score the *newly generated* text, per Sections 2-3).
    """
    text: str
    prefill: str = ""
    finish_reason: str | None = None


class ChatBackend(ABC):
    """Common surface for all model backends."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        sampling: SamplingConfig,
        n: int = 1,
        prefill: str | None = None,
    ) -> list[GenerationResult]:
        """Generate ``n`` continuations of ``messages``.

        If ``prefill`` is given, the final assistant turn is *started* with that
        text and the model continues it; the returned ``text`` excludes the
        prefill.  Backends that do not support prefill raise
        :class:`NotImplementedError` when ``prefill`` is set.
        """

    # Optional capability — only HF backends implement these.
    def supports_prefill(self) -> bool:
        return self.spec.supports_prefill

    def supports_logits(self) -> bool:
        return self.spec.supports_logits

    def close(self) -> None:  # pragma: no cover - resource cleanup hook
        pass
