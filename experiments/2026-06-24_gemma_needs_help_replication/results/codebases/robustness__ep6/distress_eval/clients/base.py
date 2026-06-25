"""Common client interface used by every backend.

A `ModelClient` exposes:
  - chat(messages, n, temperature, ...)            standard multi-turn sampling
  - complete_with_prefill(messages, prefill, ...)  force-continue an assistant turn
                                                   (local models only; Section 3/4)
  - hidden_states(messages)                        residual stream (local only; App. I)

The chat APIs (Gemini) implement only `chat`; the others raise
NotImplementedError, and the experiments that need them are gated on
`ModelSpec.supports_prefill` / `supports_hidden_states`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str       # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    # populated only where the backend returns them
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


class ModelClient:
    """Abstract base. Subclasses implement at least `chat`."""

    name: str

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> list[GenerationResult]:
        raise NotImplementedError

    # --- optional capabilities (local HF models) -------------------------- #
    def complete_with_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> list[GenerationResult]:
        """Continue the final assistant turn starting from `prefill`. Returns the
        continuation ONLY (prefill stripped), matching the paper's "score the
        generated continuation excluding prefill" protocol (Section 3.1)."""
        raise NotImplementedError(f"{type(self).__name__} does not support prefill")

    def hidden_states(self, messages: list[ChatMessage]):
        """Return per-layer residual-stream activations for the given context.
        Used by the internal-emotion probe (Appendix I)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose hidden states"
        )

    def close(self) -> None:  # pragma: no cover - resource cleanup hook
        pass
