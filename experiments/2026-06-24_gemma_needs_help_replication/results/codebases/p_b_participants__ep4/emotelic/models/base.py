"""Common chat-client interface used by every backend.

A single `LLMClient.generate(messages, ...)` contract lets the elicitation,
prefill, judge and Petri code stay backend-agnostic. Two capabilities matter
beyond plain chat:

  * prefill  — continue from a partially-written assistant turn. Essential for
    the Section 3 base-vs-instruct comparison; only the local HF backend can do
    it faithfully (most hosted APIs cannot prefill assistant text).
  * token-level truncation — needed to cut a response "20 tokens in" (Section 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationResult:
    text: str
    model: str
    finish_reason: str | None = None
    # raw token ids of the *generated* text when the backend exposes them
    # (HF local), used by the prefill experiment for token-accurate truncation.
    token_ids: list[int] | None = None
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    name: str
    supports_prefill: bool

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate a single assistant turn.

        If `prefill` is given, the assistant turn is forced to begin with that
        string and the returned text INCLUDES the prefill (callers strip it when
        they only want the continuation).
        """
        ...
