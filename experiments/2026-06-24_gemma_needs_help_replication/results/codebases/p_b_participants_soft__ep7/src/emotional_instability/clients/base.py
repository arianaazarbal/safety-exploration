"""Shared message/result types and the abstract client interface."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationResult:
    text: str
    model: str
    finish_reason: str | None = None
    # `prefill` echoes any assistant-message prefix we asked the model to continue
    # from (Section 3). `text` is ONLY the newly generated continuation, never the
    # prefill -- the paper scores "the generated continuation (excluding prefill)".
    prefill: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    stop: list[str] | None = None
    seed: int | None = None
    # When set, the model continues from this assistant-side text (prefill /
    # assistant-prefix continuation). Supported by local HF backends and by
    # OpenRouter providers that allow trailing assistant messages.
    prefill: str | None = None


class ModelClient(abc.ABC):
    """Minimal chat interface implemented by every backend."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def chat(self, messages: list[ChatMessage], params: SamplingParams) -> GenerationResult:
        """Generate a single completion for a chat-formatted conversation."""

    def chat_batch(
        self, conversations: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        """Default sequential batch; local backends override for true batching."""
        return [self.chat(c, params) for c in conversations]
