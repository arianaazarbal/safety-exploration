"""Common chat-model interface used by the evaluation harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class SampleParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024


class ChatModel(Protocol):
    """Backend-agnostic interface.

    The harness only needs to (a) take a conversation and produce `n` sampled
    continuations, and (b) for base models, continue from a prefilled assistant
    string. Both Gemma (vLLM) and Gemini implement this.
    """

    name: str
    supports_prefill: bool   # base/open models can continue arbitrary text

    def generate(
        self,
        messages: list[Message],
        n: int = 1,
        params: SampleParams | None = None,
        prefill: str | None = None,
    ) -> list[str]:
        """Return `n` assistant continuations for `messages`.

        If `prefill` is given (base-model continuation, Section 3), the returned
        strings are the continuation only (excluding the prefill text), matching
        the paper's "score the generated continuation excluding prefill".
        """
        ...
