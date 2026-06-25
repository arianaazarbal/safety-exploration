"""Common interface for evaluated target models.

A `ModelClient` must support ordinary multi-turn chat generation. Local HF
clients additionally support *prefill continuation* (continuing from a partial
assistant turn) and *hidden-state extraction*, which the Section 3 prefill
experiment and the Appendix I internal-emotion detection rely on. API clients
(Gemini) raise NotImplementedError for those — the paper does not (and cannot)
run prefill/internal experiments on the closed Gemini models either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    role: str            # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    prompt_token_count: Optional[int] = None
    completion_token_count: Optional[int] = None
    finish_reason: Optional[str] = None
    raw: dict = field(default_factory=dict)


class ModelClient:
    """Abstract base class for an evaluated model."""

    name: str
    supports_prefill: bool = False
    supports_hidden_states: bool = False

    # ----- core chat generation ----- #
    def generate(self, messages: list[ChatMessage], *,
                 temperature: float, max_new_tokens: int,
                 seed: Optional[int] = None) -> GenerationResult:
        raise NotImplementedError

    def generate_batch(self, batch: list[list[ChatMessage]], *,
                       temperature: float, max_new_tokens: int,
                       seed: Optional[int] = None) -> list[GenerationResult]:
        """Default: serial fallback. HF/vLLM clients override for throughput."""
        return [self.generate(m, temperature=temperature,
                              max_new_tokens=max_new_tokens, seed=seed)
                for m in batch]

    # ----- prefill continuation (Section 3) ----- #
    def continue_prefill(self, messages: list[ChatMessage], assistant_prefix: str, *,
                         temperature: float, max_new_tokens: int,
                         seed: Optional[int] = None) -> GenerationResult:
        """Continue generating from `assistant_prefix` as the start of the final
        assistant turn. Returns ONLY the newly generated continuation (excluding
        the prefix), matching the paper's scoring of "the generated continuation
        (excluding prefill)"."""
        raise NotImplementedError(f"{self.name} does not support prefill continuation")

    # ----- token utilities (needed for the 20-token "early" truncation) ----- #
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError
