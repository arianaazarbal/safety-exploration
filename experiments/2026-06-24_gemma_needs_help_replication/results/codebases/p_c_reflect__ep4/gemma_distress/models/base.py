"""Abstract interface shared by every target-model client.

A conversation is a list of :class:`Turn`. Clients implement two operations:

* :meth:`respond` -- given a conversation, produce the next assistant turn.
* :meth:`continue_prefill` -- given a conversation whose last turn is a partial
  assistant message, continue *that* message. Used by the Section 3 prefill
  experiment. Only open-weight backends support this.

Generation defaults to temperature 1 (Section 2.1).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal, Sequence

from gemma_distress import config

Role = Literal["system", "user", "assistant"]


@dataclass
class Turn:
    role: Role
    content: str


@dataclass
class GenerationParams:
    temperature: float = config.SAMPLING_TEMPERATURE
    max_new_tokens: int = config.MAX_NEW_TOKENS
    # Whether to request/allow internal "thinking". The paper sets thinking to
    # false for all models (Appendix B.1); we mirror that default.
    thinking: bool = False
    # Stop generation early at these strings, when the backend supports it.
    stop: tuple[str, ...] = field(default_factory=tuple)


class ModelClient(abc.ABC):
    """Common surface for Gemma (local) and Gemini (API) clients."""

    spec: "config.ModelSpec"

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def supports_prefill(self) -> bool:
        return self.spec.supports_prefill

    @abc.abstractmethod
    def respond(
        self,
        conversation: Sequence[Turn],
        params: GenerationParams | None = None,
    ) -> str:
        """Return the assistant's reply to ``conversation``."""

    def continue_prefill(
        self,
        conversation: Sequence[Turn],
        prefill: str,
        params: GenerationParams | None = None,
    ) -> str:
        """Continue a prefilled assistant turn.

        ``conversation`` ends with the user turn; ``prefill`` is the partial
        assistant text the model should continue from. Returns the *generated
        continuation only* (excluding the prefill), matching the paper's
        scoring convention (Section 3.1).
        """
        raise NotImplementedError(
            f"{self.spec.name} ({self.spec.backend}) does not support prefilling."
        )

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        pass
