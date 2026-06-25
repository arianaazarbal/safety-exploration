"""Abstract chat-model interface shared by every backend.

A ``Message`` is a ``{"role": "user"|"assistant"|"system", "content": str}`` dict
(OpenAI-style). All experiments speak to models exclusively through
``ChatModel.generate`` / ``generate_batch`` so that local Gemma checkpoints and
the OpenRouter Gemini endpoints are interchangeable.

Base (pretrained, non-chat) Gemma models implement the same interface but ignore
chat structure: see ``hf_model.HFCompletionModel`` which exposes ``complete`` for
the prefill experiment (Section 3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class Message(TypedDict):
    role: str       # "system" | "user" | "assistant"
    content: str


class ChatModel(ABC):
    """A multi-turn chat model."""

    name: str

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        """Return a single assistant completion for ``messages``."""

    def generate_batch(
        self,
        batch: list[list[Message]],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seeds: list[int] | None = None,
    ) -> list[str]:
        """Default sequential batch; HF backend overrides for true batching."""
        seeds = seeds or [None] * len(batch)
        return [
            self.generate(
                m,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=s,
            )
            for m, s in zip(batch, seeds)
        ]


class CompletionModel(ABC):
    """A raw text-completion (base / pretrained) model used for prefilling."""

    name: str

    @abstractmethod
    def complete(
        self,
        prefix: str,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
        seed: int | None = None,
    ) -> str:
        """Continue ``prefix`` and return *only* the newly generated text."""
