"""Common model-client interface.

Every backend (local Gemma, remote Gemini) implements `ModelClient`. The harness
only ever talks to this interface, so swapping backends or adding LoRA finetunes
requires no change in the eval/training code.

Two generation modes are needed by the paper:

1. `chat(messages)` — standard multi-turn chat completion. Used by the Section 2
   elicitation rollouts and Petri.
2. `continue_prefill(messages, prefill)` — append `prefill` to the start of the
   assistant turn and let the model *continue* it. Used by the Section 3
   base-vs-instruct experiment and the Section 4 recovery experiment, where base
   models (untrained on chat format) must be coaxed to continue from a fixed
   starting point. Only local (HF) backends can truly prefill; API backends
   approximate it (see openrouter_model.py).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenerationResult:
    """A single completion plus optional decoding metadata."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # token-id sequence of the completion (HF backend only) — used by the
    # prefill experiment to truncate "20 tokens into the turn" precisely.
    completion_token_ids: list[int] | None = None
    finish_reason: str | None = None
    meta: dict = field(default_factory=dict)


class ModelClient(abc.ABC):
    """Abstract inference client. Backends must be safe to call concurrently
    at the granularity the runner uses (it serialises per-client for HF, and
    fans out with a thread pool for API backends)."""

    def __init__(self, spec) -> None:  # spec: config.ModelSpec
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @abc.abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate the assistant's next turn given a conversation."""

    @abc.abstractmethod
    def continue_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> GenerationResult:
        """Continue an assistant turn that is pre-seeded with `prefill`.

        Returns ONLY the continuation (the prefill is stripped), matching the
        paper: "The generated continuation (excluding prefill) is scored".
        """

    def supports_logits(self) -> bool:
        """True if `residual_logits` is implemented (Appendix I probing)."""
        return False

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        pass
