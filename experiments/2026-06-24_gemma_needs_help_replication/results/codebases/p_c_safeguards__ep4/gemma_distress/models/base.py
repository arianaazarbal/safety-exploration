"""Model client interface shared by local and API backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict


class Message(TypedDict):
    role: str        # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    top_p: float = 1.0
    stop: list[str] | None = None
    # `prefill` forces the assistant turn to begin with this text and the model
    # continues from it. Only supported by local backends (Section 3 / recovery).
    prefill: str | None = None


class ModelClient(Protocol):
    """Minimal contract every backend implements."""

    spec: Any  # ModelSpec; typed as Any to avoid an import cycle

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        """Return the assistant completion text for a chat conversation.

        If ``cfg.prefill`` is set, the returned text EXCLUDES the prefill (only
        the model's continuation is returned), matching how the prefill
        experiments score "the generated continuation (excluding prefill)".
        """
        ...

    def supports_prefill(self) -> bool:
        ...

    def close(self) -> None:
        ...
