"""Backend abstraction over the (very different) ways we reach models.

A single ``ChatBackend`` interface covers:
  * local Gemma via vLLM or transformers (with assistant *prefill* support,
    required for Section 3),
  * Gemini via OpenRouter,
  * Claude / GPT via their APIs (judging, auditing, paraphrasing).

The two capabilities every backend must expose:
  * ``generate``          -- standard chat completion from a message list.
  * ``generate_prefill``  -- continue a *prefilled* assistant turn. Only local
                             backends implement this; API backends raise.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypedDict

from ..config import GenConfig


class ChatMessage(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class GenResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: object | None = None


class ChatBackend(abc.ABC):
    """Common interface for every model we talk to."""

    supports_prefill: bool = False

    def __init__(self, spec, **kwargs):
        self.spec = spec
        self.model_id = spec.model_id

    @abc.abstractmethod
    def generate(self, messages: list[ChatMessage], gen: GenConfig) -> GenResult:
        """Single chat completion."""

    def generate_batch(self, batch: list[list[ChatMessage]], gen: GenConfig) -> list[GenResult]:
        """Default: serial fallback. vLLM/API backends override for throughput."""
        return [self.generate(m, gen) for m in batch]

    def generate_prefill(
        self, messages: list[ChatMessage], prefill: str, gen: GenConfig
    ) -> GenResult:
        """Continue an assistant turn that already starts with ``prefill``.

        Returns ONLY the newly generated continuation (excluding the prefill),
        matching the paper's convention in Section 3.1 ("The generated
        continuation (excluding prefill) is scored").
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support assistant prefill; use a local backend."
        )

    def close(self) -> None:
        pass
