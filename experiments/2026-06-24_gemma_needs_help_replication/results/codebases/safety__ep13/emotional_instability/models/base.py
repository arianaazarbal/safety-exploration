"""Uniform chat/generation interface shared by every backend.

A ``ModelClient`` exposes two operations the experiments need:

* :meth:`generate` -- standard multi-turn chat completion.
* :meth:`generate_with_prefill` -- continue from a forced assistant prefix
  (Section 3). This is the operation base models need to behave consistently;
  instruct models support it too so the comparison is apples-to-apples.

Messages use a tiny role/content record rather than any provider-specific type.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class GenerationResult:
    text: str
    # Token ids of the *generated* continuation only (excludes prompt/prefill).
    # Populated by local backends; None for API backends.
    new_token_ids: list[int] | None = field(default=None)
    finish_reason: str | None = None


class ModelClient(abc.ABC):
    """Abstract chat model."""

    def __init__(self, spec) -> None:  # spec: config.ModelSpec
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @abc.abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        ...

    def generate_with_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Continue an assistant turn that already starts with ``prefill``.

        The returned text is the *continuation only* (excluding ``prefill``).
        Backends that cannot truly prefill (most hosted APIs) should raise
        ``NotImplementedError`` so callers can route base-model work to local
        backends.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support response prefilling")

    # Batched convenience. Backends may override with a faster implementation.
    def generate_batch(
        self,
        batch: list[list[ChatMessage]],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[GenerationResult]:
        return [
            self.generate(m, temperature=temperature,
                          max_new_tokens=max_new_tokens)
            for m in batch
        ]

    def close(self) -> None:  # release GPU memory etc.
        pass
