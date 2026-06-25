"""Backend interface shared by all model-serving implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    role: str        # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationRequest:
    """A single generation request.

    `messages` is the conversation so far (ending on a user turn for a normal
    chat request). If `prefill` is set, the assistant turn is *seeded* with
    that text and the model continues from it - this is how Section 3 forces
    base models to continue a partially written response, and how we grade only
    the model-generated continuation.
    """

    messages: list[Message]
    n: int = 1
    temperature: float = 1.0
    max_tokens: int = 2048
    prefill: str | None = None
    stop: list[str] | None = None
    seed: int | None = None


@runtime_checkable
class ChatBackend(Protocol):
    """Minimal interface every backend implements.

    Implementations should be safe to call concurrently from threads (the API
    backends) or to receive large batches (the vLLM backend batches internally).
    """

    spec_name: str
    supports_prefill: bool

    def generate(self, request: GenerationRequest) -> list[str]:
        """Return `request.n` completions (assistant text only).

        When `request.prefill` is provided, the returned strings are the
        *continuation only* (the prefill is stripped), matching the paper's
        "score the model-generated continuation, excluding the prefilled text".
        """
        ...

    def generate_batch(self, requests: list[GenerationRequest]) -> list[list[str]]:
        """Vectorised `generate`. Default impls may loop; vLLM batches."""
        ...
