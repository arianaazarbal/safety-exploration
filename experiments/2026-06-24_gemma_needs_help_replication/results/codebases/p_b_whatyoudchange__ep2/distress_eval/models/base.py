"""Abstract model interface shared by all target/judge backends.

A `ModelClient` is the single seam the rest of the codebase talks to. It must be
able to (a) continue a multi-turn chat, and optionally (b) *prefill* an assistant
turn — i.e. force the model to continue from a given partial assistant string,
which the base-vs-instruct prefill experiment (Section 3) and the recovery
experiment (Section 4) both depend on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_tokens: int = 2048
    # When set, the model continues *from* this text as the start of its
    # assistant turn rather than starting fresh. The returned text EXCLUDES the
    # prefill (callers that need the full turn should re-concatenate).
    prefill: str | None = None
    # Disable any provider-side hidden reasoning where supported (Appendix B.1:
    # "we set thinking to be false via the API").
    thinking: bool = False
    stop: Sequence[str] | None = None


class ModelClient:
    """Backend-agnostic generation interface."""

    spec_name: str
    supports_prefill: bool = False

    def generate(
        self, messages: Sequence[ChatMessage], cfg: GenerationConfig
    ) -> str:
        """Return the assistant continuation (prefill excluded) for `messages`."""
        raise NotImplementedError

    def generate_batch(
        self, batch: Sequence[Sequence[ChatMessage]], cfg: GenerationConfig
    ) -> list[str]:
        """Default: serial fallback. Local backends override for true batching."""
        return [self.generate(m, cfg) for m in batch]

    # Optional: only implemented by local HF backends (needed for Appendix I).
    def forward_with_hidden_states(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            f"{self.spec_name} backend does not expose hidden states"
        )
