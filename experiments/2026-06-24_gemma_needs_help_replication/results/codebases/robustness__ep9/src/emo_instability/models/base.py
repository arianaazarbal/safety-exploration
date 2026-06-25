"""Shared model-client interface and message types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from ..config import SamplingConfig

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class GenerationError(RuntimeError):
    """Raised when a backend fails to produce a generation after retries."""


class ModelClient(ABC):
    """Minimal interface every backend implements.

    The elicitation harness only needs batched chat generation; the prefill
    experiment additionally needs raw-text continuation (``complete`` /
    ``continue_chat``), which API backends do not support.
    """

    spec_key: str

    @abstractmethod
    def generate(self, messages: list[ChatMessage], sampling: SamplingConfig) -> str:
        """Generate one assistant turn given a chat history."""

    def generate_batch(
        self, batch: list[list[ChatMessage]], sampling: SamplingConfig
    ) -> list[str]:
        """Generate for many chat histories. Default: sequential ``generate``.

        Backends that support true batching (vLLM) override this.
        """
        return [self.generate(m, sampling) for m in batch]

    # -- optional capabilities (base-model prefill experiment) --------------
    def supports_completion(self) -> bool:
        return False

    def complete(self, prompt_text: str, sampling: SamplingConfig) -> str:
        """Raw text continuation from a string prompt (base models)."""
        raise NotImplementedError(f"{type(self).__name__} does not support completion")

    def complete_batch(
        self, prompts: list[str], sampling: SamplingConfig
    ) -> list[str]:
        return [self.complete(p, sampling) for p in prompts]

    def continue_chat(
        self, messages: list[ChatMessage], prefill: str, sampling: SamplingConfig
    ) -> str:
        """Continue an assistant turn that is *prefilled* with ``prefill`` text.

        Used by the Section 3 prefill experiment: the model continues from a fixed
        emotional/neutral start rather than generating from scratch.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support prefill continuation")
