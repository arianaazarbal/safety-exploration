"""Common interface for target models.

A ``Message`` is ``{"role": "user"|"assistant"|"system", "content": str}``.
``ChatModel`` is the abstraction the eval/prefill/training code talks to, so the
rest of the codebase never branches on Gemma-vs-Gemini or local-vs-API.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


@dataclass
class GenConfig:
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None


class ChatModel(abc.ABC):
    """A model we can sample multi-turn chat continuations from."""

    #: Whether ``continue_prefill`` is supported. True for local HF/vLLM models
    #: (we control the prompt), False for chat-only API models like Gemini.
    supports_prefill: bool = False

    def __init__(self, name: str, is_base: bool = False):
        self.name = name
        self.is_base = is_base

    @abc.abstractmethod
    def generate(self, messages: list[Message], cfg: GenConfig) -> str:
        """Return one assistant completion for ``messages``."""

    def generate_batch(
        self, batch: list[list[Message]], cfg: GenConfig
    ) -> list[str]:
        """Default: sequential. Backends override for true batching."""
        return [self.generate(m, cfg) for m in batch]

    def continue_prefill(
        self, messages: list[Message], prefill: str, cfg: GenConfig
    ) -> str:
        """Continue an assistant turn that *begins with* ``prefill``.

        Returns only the newly generated text (excluding ``prefill``). Used by
        the base-vs-instruct prefill experiment (Sec 3). Only local backends
        implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled continuations"
        )

    def continue_prefill_batch(
        self, batch: list[tuple[list[Message], str]], cfg: GenConfig
    ) -> list[str]:
        return [self.continue_prefill(m, p, cfg) for m, p in batch]

    def close(self) -> None:  # free GPU memory etc.
        pass
