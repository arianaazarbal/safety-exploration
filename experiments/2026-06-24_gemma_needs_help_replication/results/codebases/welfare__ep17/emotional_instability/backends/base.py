"""Backend interface shared by HF-local and OpenRouter models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str          # "user" | "assistant" | "system"
    content: str


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    top_p: float = 1.0
    # Continuation control for the prefill experiment: when `prefill` is set, the
    # backend appends it to the (rendered) prompt and generates a continuation,
    # returning ONLY the newly generated text (excluding the prefill).
    prefill: str | None = None


class ChatBackend(Protocol):
    """Minimal surface every backend implements.

    `chat` runs a normal multi-turn completion. `chat_prefilled` continues from
    a partial assistant turn — only HF backends support this (base models need
    it); OpenRouter raises NotImplementedError.
    """

    spec_name: str
    supports_prefill: bool

    def chat(self, messages: list[Message], gen: GenConfig) -> str: ...

    def chat_prefilled(
        self, messages: list[Message], prefill: str, gen: GenConfig
    ) -> str:
        """Continue `prefill` as the assistant's reply; return continuation only."""
        ...
