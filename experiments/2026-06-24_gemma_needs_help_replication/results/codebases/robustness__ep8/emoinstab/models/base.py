"""Model-client abstraction shared by all backends.

Every experiment talks to models through :class:`ModelClient`. The two
operations that matter:

- ``chat`` / ``chat_batch``: ordinary multi-turn generation (Sections 2 & 4).
- ``continue_prefill``: continue a *partially written* assistant turn from a
  fixed prefix (Section 3 prefill experiment, Section 4.2 recovery test). Only
  the local backends implement this; API backends raise NotImplementedError.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Sequence

from emoinstab.config import ModelSpec


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


Conversation = list[Message]


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 2048
    top_p: float = 1.0
    n: int = 1
    stop: tuple[str, ...] = field(default_factory=tuple)
    seed: int | None = None
    thinking: bool = False

    def with_(self, **kw) -> "SamplingParams":
        return replace(self, **kw)


class ModelClient(ABC):
    """Uniform interface over a single model."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def default_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.spec.temperature,
            max_tokens=self.spec.max_tokens,
            thinking=self.spec.thinking,
        )

    @abstractmethod
    def chat(self, messages: Conversation, params: SamplingParams | None = None) -> list[str]:
        """Return ``params.n`` completions for one conversation."""

    def chat_batch(
        self,
        conversations: Sequence[Conversation],
        params: SamplingParams | None = None,
    ) -> list[list[str]]:
        """Default batched implementation = loop over ``chat``.

        Local backends (vLLM / HF) override this for real batching.
        """
        return [self.chat(c, params) for c in conversations]

    def continue_prefill(
        self,
        messages: Conversation,
        prefill: str,
        params: SamplingParams | None = None,
    ) -> list[str]:
        """Continue an assistant turn that already starts with ``prefill``.

        Returns only the *newly generated* continuation (excluding ``prefill``),
        matching the paper's "score the continuation, excluding the prefill"
        protocol (Section 3.1).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefill continuation."
        )

    def close(self) -> None:  # pragma: no cover - backend cleanup hook
        pass
