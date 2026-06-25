"""Common chat-model interface.

A single abstraction sits in front of both inference paths:

- local HuggingFace Gemma (supports assistant *prefill* and hidden-state capture),
- OpenRouter-hosted Gemini (chat completions only).

Everything downstream (rollouts, prefill experiment, judge) talks to this
interface so the experiment code never branches on provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class GenerationResult:
    """One sampled completion plus optional token-level metadata."""

    text: str
    # token ids of the *generated* continuation only (excludes any prefill);
    # populated by local backends, may be None for API backends.
    token_ids: list[int] | None = None
    finish_reason: str | None = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class ChatModel(Protocol):
    """Minimal interface every backend implements."""

    spec_name: str

    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        n: int = 1,
        assistant_prefill: str | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> list[GenerationResult]:
        """Sample ``n`` completions for ``messages``.

        ``assistant_prefill`` forces the model to continue from the given text
        (Section 3). Backends that cannot honour a prefill must raise
        :class:`PrefillNotSupported`.
        """
        ...

    @property
    def supports_prefill(self) -> bool:
        ...


class PrefillNotSupported(RuntimeError):
    """Raised when a backend cannot continue from an assistant prefill."""
