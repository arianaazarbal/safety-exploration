"""Common model-client interface and message types."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class GenerationResult:
    text: str
    # Optional captured internals (HF backend only).
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    # Layer x token residual stream, captured only when requested (probing).
    hidden_states: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ModelClient(abc.ABC):
    """Uniform access to a single model.

    Implementations must be safe to call repeatedly. ``n`` lets a single call
    request multiple independent samples (at temperature 1 this is how we draw
    the many rollouts the paper requires).
    """

    spec: Any  # config.ModelSpec — typed loosely to avoid an import cycle.

    @abc.abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[GenerationResult]:
        """Generate ``n`` completions for a chat-formatted conversation."""

    # --- Optional capabilities (open-weight models only) -------------------- #

    def supports_prefill(self) -> bool:
        return False

    def continue_prefill(
        self,
        messages: Sequence[ChatMessage],
        prefill: str,
        *,
        n: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        capture_hidden_states: bool = False,
    ) -> list[GenerationResult]:
        """Continue a partially-written assistant turn (the ``prefill``).

        Used by Section 3 (base vs instruct continuations) and Section 4.2
        (recovery-from-distress prefills). Only the generated continuation —
        not the prefill — is returned in ``text``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support prefill")

    def supports_hidden_states(self) -> bool:
        return False
