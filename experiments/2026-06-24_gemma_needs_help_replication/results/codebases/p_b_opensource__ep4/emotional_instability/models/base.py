"""Backend-agnostic types and the `ModelBackend` interface.

The whole harness speaks in `ChatMessage` lists. Two generation modes are
supported:

* `generate` — standard chat completion. Used for all instruct models and all
  API models in Sections 2 and 4.
* `generate_with_prefill` — force the assistant turn to *begin* with a given
  string and return only the continuation. This is the mechanism the paper uses
  to compare base and instruct models on equal footing (Section 3), and to
  measure recovery from high-frustration states (Section 4.2). Base (pretrained)
  models are *only* ever called this way.

Backends should implement batched variants where the underlying engine supports
it; the default batch implementations fall back to sequential calls.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal, Optional

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationResult:
    """A single sampled completion.

    `text` is the assistant's reply. For prefilled generations it excludes the
    prefill (the paper scores "the generated continuation (excluding prefill)").
    `prefill` records what was forced, for provenance.
    """

    text: str
    prefill: Optional[str] = None
    finish_reason: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    seed: Optional[int] = None
    stop: Optional[list[str]] = None


class ModelBackend(abc.ABC):
    """Common interface every backend implements."""

    #: True if this backend can force-continue an assistant prefix.
    supports_prefill: bool = False
    #: True for pretrained (non-chat) checkpoints.
    is_base: bool = False

    @abc.abstractmethod
    def generate(
        self, messages: list[ChatMessage], params: SamplingParams
    ) -> GenerationResult:
        """Sample one assistant completion for `messages`."""

    def generate_batch(
        self, batch: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        """Sample one completion per conversation. Default: sequential."""
        return [self.generate(m, params) for m in batch]

    def generate_with_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        params: SamplingParams,
    ) -> GenerationResult:
        """Force the assistant turn to start with `prefill`; return the
        continuation only. Backends that cannot do this raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled generation"
        )

    def generate_with_prefill_batch(
        self,
        batch: list[tuple[list[ChatMessage], str]],
        params: SamplingParams,
    ) -> list[GenerationResult]:
        return [
            self.generate_with_prefill(m, p, params) for (m, p) in batch
        ]
