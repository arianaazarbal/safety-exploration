"""Backend-agnostic chat-model interface."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypedDict

from ..config import ModelSpec


class Message(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    # Filled when the backend can report it (hf can; gemini usually can't).
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatModel(abc.ABC):
    """Minimal multi-turn chat interface used by the distress harness."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.key = spec.key
        self.display = spec.display

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        prefill: str | None = None,
    ) -> GenerationResult:
        """Generate one assistant turn.

        ``messages`` is the full conversation so far (system optional, then
        alternating user/assistant). ``prefill`` is an optional assistant
        prefix the model must continue from verbatim — used in Section 3 to
        force base models to continue from a fixed emotional trajectory. The
        returned ``text`` is the continuation only (it does NOT include the
        prefill); callers that need the full turn concatenate themselves.
        """

    def supports_prefill(self) -> bool:
        return False
