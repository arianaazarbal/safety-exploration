"""Backend interface shared by subject models (Gemma local, Gemini API)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    # Number of newly generated tokens (when the backend can report it).
    n_new_tokens: int | None = None
    # Raw provider metadata (finish reason, hidden-reasoning flags, etc.).
    meta: dict | None = None


class ModelBackend(Protocol):
    """Minimal interface every subject backend implements.

    `generate` runs a normal chat completion. `continue_text` performs prefilled
    continuation (used by the Section 3 prefill experiment and by base models):
    the model continues from `prefill` as the start of its assistant turn,
    rather than starting a fresh turn. The returned text EXCLUDES the prefill.
    """

    name: str
    supports_chat: bool
    supports_prefill: bool

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> GenerationResult: ...

    def continue_text(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> GenerationResult: ...
