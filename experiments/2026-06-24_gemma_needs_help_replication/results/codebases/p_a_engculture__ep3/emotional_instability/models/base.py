"""Model-client abstraction.

The replication touches four kinds of model access:

  * vLLM         — high-throughput temperature-1 sampling of Gemma instruct
                   weights for the Section 2 / Section 4 evaluations.
  * HF transformers — explicit assistant *prefill* and hidden-state access, for
                   the Section 3 base-vs-instruct continuations and the
                   Appendix I logit-lens probing.
  * OpenRouter   — Gemini (and, optionally, other closed models) via an
                   OpenAI-compatible chat endpoint.
  * Anthropic    — the Claude judge / Petri auditor+judge.

All four expose the same minimal surface: ``generate`` for a chat conversation,
and (for local backends) ``generate_with_prefill`` for forcing the start of the
assistant turn. Keeping the surface this small is what lets the rollout engine
and the judge be backend-agnostic.
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
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 2048
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    thinking: bool = False           # disable hidden reasoning where supported
    seed: int | None = None          # per-sample seed; many backends ignore


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw: object | None = None        # provider-native response, for debugging


@runtime_checkable
class ModelClient(Protocol):
    """Minimal generation interface implemented by every backend."""

    name: str

    def generate(self, messages: list[ChatMessage], params: SamplingParams) -> GenerationResult:
        ...

    def generate_batch(
        self, conversations: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        """Default: sequential. vLLM overrides this with true batching."""
        ...


class PrefillMixin:
    """Optional capability: continue a forced assistant prefix.

    Used by the Section 3 prefill experiment. API backends (Gemini) cannot do
    this faithfully, so only the local backends implement it.
    """

    def generate_with_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        params: SamplingParams,
    ) -> GenerationResult:  # pragma: no cover - interface only
        raise NotImplementedError
