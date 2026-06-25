"""Common chat-model interface shared by all backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    finish_reason: str | None = None
    raw: dict | None = None


class ChatModel(Protocol):
    """Minimal interface every backend implements.

    ``generate`` runs a single completion given a chat history. ``prefill`` (only
    meaningful for local HF models) forces the assistant to *continue* a given
    partial response and returns just the continuation; closed API models raise
    NotImplementedError.
    """

    key: str
    supports_prefill: bool

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        ...

    def prefill(
        self,
        messages: list[Message],
        prefill_text: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        ...
