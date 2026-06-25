"""Backend abstraction shared by every model the harness talks to.

Three operations cover all experiments:
  * ``chat``     — standard multi-turn chat completion (eval rollouts, judge, auditor).
  * ``continue_from`` — prefill: given a conversation, force the assistant to *continue*
                    from a fixed prefix and return only the continuation (§3).
  * ``score_logits`` — return per-token next-token logits / residual stream for the
                    internal-emotion probe (App. I). Only the local HF backend supports it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import ModelSpec
from ..utils import Message


class GenerationError(RuntimeError):
    """Raised when a backend cannot produce a completion (after retries)."""


class ModelBackend(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant completion for ``messages``."""

    def continue_from(
        self,
        messages: list[Message],
        prefix: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Continue an assistant turn from ``prefix``; return ONLY the new text.

        Required for the §3 prefill experiments. API backends cannot truly prefill, so they
        raise NotImplementedError; only the local HF backend overrides this.
        """
        raise NotImplementedError(
            f"Backend for {self.spec.name} does not support prefilling (continue_from)."
        )

    def supports_prefill(self) -> bool:
        return type(self).continue_from is not ModelBackend.continue_from

    def _temperature(self, override: float | None) -> float:
        return self.spec.temperature if override is None else override

    def _max_tokens(self, override: int | None) -> int:
        return self.spec.max_tokens if override is None else override
