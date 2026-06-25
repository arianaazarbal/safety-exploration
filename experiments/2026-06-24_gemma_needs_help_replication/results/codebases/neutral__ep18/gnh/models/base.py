"""Unified model-backend interface.

All sampling in the project goes through `ModelBackend.generate`. A backend is
either a local HuggingFace model (Gemma) or a remote API (Gemini via OpenRouter,
Claude via Anthropic). Keeping a single interface means the rollout / judge /
Petri code is agnostic to where a model lives.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, TypedDict


class ChatMessage(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    # Tokenised length of the generated text (best-effort; APIs report usage).
    n_tokens: int | None = None
    finish_reason: str | None = None


class ModelBackend(ABC):
    """A thing we can sample completions from, given a chat transcript."""

    family: str
    kind: str  # "instruct" | "base"

    @abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[GenerationResult]:
        """Return `n` completions for `messages`.

        `prefill` is an assistant-turn prefix the model must continue from (used
        for the base-vs-instruct experiment in Section 3, and supported by the
        Anthropic API for instruct models). The returned text EXCLUDES the
        prefill -- callers that want the full assistant turn must concatenate.
        """

    def count_tokens(self, text: str) -> int:
        """Best-effort token count. Backends override with their tokenizer."""
        # crude fallback: ~4 chars/token
        return max(1, len(text) // 4)
