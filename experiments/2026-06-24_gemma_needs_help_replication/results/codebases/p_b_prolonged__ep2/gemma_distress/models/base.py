"""Abstract target-model interface shared by the HF (Gemma) and Gemini backends.

A conversation is a list of ``ChatTurn`` dicts ``{"role": "user"|"assistant",
"content": str}``. ``system`` content, when used, is passed separately.

Two capabilities matter for the paper:

* ``chat`` -- standard multi-turn completion (all backends).
* ``continue_prefill`` -- continue from a *prefilled* assistant response
  (Section 3 / recovery). Only white-box (HF) backends support this; API
  backends raise ``PrefillUnsupported``.
"""
from __future__ import annotations

import abc
from typing import Optional, TypedDict

from ..config import ModelSpec, RunConfig, SamplingConfig


class ChatTurn(TypedDict):
    role: str       # "user" | "assistant"
    content: str


class PrefillUnsupported(RuntimeError):
    """Raised when a prefill continuation is requested from an API-only model."""


class TargetBackend(abc.ABC):
    def __init__(self, spec: ModelSpec, cfg: RunConfig):
        self.spec = spec
        self.cfg = cfg

    # -- required --------------------------------------------------------
    @abc.abstractmethod
    def chat(self, messages: list[ChatTurn], sampling: SamplingConfig,
             system: Optional[str] = None) -> str:
        """Return a single assistant completion for the given conversation."""

    # -- optional (white-box only) --------------------------------------
    def supports_prefill(self) -> bool:
        return False

    def continue_prefill(self, messages: list[ChatTurn], prefill: str,
                         sampling: SamplingConfig, n: int = 1,
                         system: Optional[str] = None) -> list[str]:
        """Continue from a prefilled final assistant turn `prefill`, returning
        `n` continuations (the prefill text itself is NOT included in the
        returned strings). White-box backends override this."""
        raise PrefillUnsupported(
            f"Model '{self.spec.name}' ({self.spec.backend}) does not support "
            f"prefill continuation."
        )

    def close(self) -> None:
        """Release any held resources (GPU memory, sessions). Optional."""
