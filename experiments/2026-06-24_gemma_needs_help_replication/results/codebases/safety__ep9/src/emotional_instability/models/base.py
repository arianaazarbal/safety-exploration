"""Unified model-client interface.

All experiments talk to models through `ModelClient`. Two backends implement it:
`HFModelClient` (local Gemma via vLLM, supports prefilling, base models, and
LoRA adapters) and `APIModelClient` (Gemini via an OpenAI-compatible endpoint).

The interface is batch-first because the evals sample thousands of responses and
the per-prefill continuation experiment needs n=50 samples from one context.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


Conversation = list[Message]


class ModelClient(ABC):
    """A model that can produce completions for batches of conversations."""

    name: str
    supports_prefill: bool = False
    supports_raw: bool = False

    @abstractmethod
    def generate(
        self,
        conversations: list[Conversation],
        *,
        n: int = 1,
        prefill: list[str] | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[list[str]]:
        """Return, for each conversation, a list of `n` completions.

        `prefill` (one string per conversation, or None) seeds the assistant
        turn; the returned text is the *continuation only* (prefill stripped).
        Backends that do not support prefilling raise if it is provided.
        """

    def generate_raw(
        self,
        prompts: list[str],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[list[str]]:
        """Raw text continuation with no chat template (base/pretrained models)."""
        raise NotImplementedError(f"{type(self).__name__} does not support raw completion")

    def close(self) -> None:  # pragma: no cover - backend specific
        pass
