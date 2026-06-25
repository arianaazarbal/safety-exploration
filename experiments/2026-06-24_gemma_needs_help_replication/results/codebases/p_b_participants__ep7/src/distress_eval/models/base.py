"""Abstract model-client interface shared by all backends.

A ``Message`` is a plain ``{"role", "content"}`` dict (roles: ``system``,
``user``, ``assistant``). All clients accept the same message list so the
elicitation harness is backend-agnostic.

Two generation modes matter for this paper:

* ``generate`` -- ordinary chat completion (Section 2 rollouts, judging).
* ``continue_from`` -- *prefill* continuation, where the model continues an
  assistant turn whose opening tokens are fixed. Section 3 relies on this to
  compare base vs instruct models from identical starting points.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, TypedDict


class Message(TypedDict):
    role: str
    content: str


@dataclass
class GenerationResult:
    text: str
    # Optional metadata: token count, finish reason, the prefill used, etc.
    meta: dict[str, Any] = field(default_factory=dict)


class ModelClient(abc.ABC):
    """Common surface for participant and tool models."""

    def __init__(self, model_id: str, *, temperature: float = 1.0, max_tokens: int = 2048,
                 options: dict[str, Any] | None = None):
        self.model_id = model_id
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.options = options or {}

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> list[GenerationResult]:
        """Return ``n`` independent completions for ``messages``."""

    def continue_from(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
        seed: int | None = None,
    ) -> list[GenerationResult]:
        """Continue an assistant turn that begins with ``prefill``.

        Returns only the *generated continuation* (excluding the prefill text),
        matching the paper's "score the continuation, excluding prefill" rule.

        Default implementation raises; backends that support prefill override
        it. (API chat models without prefill support cannot be used in
        Section 3 -- this is true of Gemini, which we document as a gap.)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefill/continuation. "
            "Section 3 (base-vs-instruct prefilling) requires open-weight access."
        )

    @property
    def supports_prefill(self) -> bool:
        return type(self).continue_from is not ModelClient.continue_from
