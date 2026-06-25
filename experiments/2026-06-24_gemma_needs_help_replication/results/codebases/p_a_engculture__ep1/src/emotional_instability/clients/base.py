"""Client interface shared by all backends.

A conversation is a list of :class:`ChatMessage`. Backends implement two
operations:

* :meth:`ModelClient.generate` — sample assistant continuations for a chat
  conversation (used by the eval harness, calm-data generation, Petri, ...).
* :meth:`ModelClient.continue_prefill` — continue from an explicit assistant
  prefix (used by the Section 3 / recovery prefilling studies). Only local
  backends support this; API backends raise :class:`PrefillUnsupported`.

The split keeps the common path simple while making the prefill requirement
explicit at the type level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationConfig:
    """Sampling parameters. Defaults follow the paper (temperature 1)."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    n: int = 1                      # number of independent samples
    seed: int | None = None
    stop: list[str] = field(default_factory=list)


class PrefillUnsupported(NotImplementedError):
    """Raised when an API-only backend is asked to continue an assistant prefill."""


class ModelClient(ABC):
    """Backend-agnostic generation interface."""

    def __init__(self, spec):
        self.spec = spec
        self.name = spec.name

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        cfg: GenerationConfig,
        system: str | None = None,
    ) -> list[str]:
        """Return ``cfg.n`` assistant completions for ``messages``.

        ``system`` is an optional system prompt (kept separate from ``messages``
        so backends that special-case the system role — Anthropic, Gemma chat
        templates — can place it correctly).
        """

    def continue_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        cfg: GenerationConfig,
        system: str | None = None,
    ) -> list[str]:
        """Continue generation from an assistant ``prefill`` string.

        The returned strings are *only the newly generated text* (the prefill is
        stripped), matching the paper's "score the continuation, excluding the
        prefilled text" protocol. Local backends override this; API backends
        leave the default, which raises.
        """
        raise PrefillUnsupported(
            f"Backend '{self.spec.backend}' for model '{self.name}' cannot "
            "continue an assistant prefill. Prefill experiments (Section 3, "
            "recovery, internal-emotion) require a local (hf) backend."
        )

    def supports_prefill(self) -> bool:
        return False
