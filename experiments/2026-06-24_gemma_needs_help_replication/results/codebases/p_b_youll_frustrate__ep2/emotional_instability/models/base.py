"""Provider interface shared by Gemma and Gemini backends."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from ..config import ModelSpec, SamplingConfig


@dataclass
class ChatMessage:
    role: str            # "system" | "user" | "assistant"
    content: str


class ModelProvider(abc.ABC):
    """A uniform chat-completion surface over one target model.

    The harness only needs ``generate``; the prefilling experiment (Section 3)
    additionally needs ``continue_from`` to make a model continue a fixed prefix
    (chat-formatted for instruct models, raw text for base models).
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abc.abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        sampling: SamplingConfig,
        seed: Optional[int] = None,
    ) -> str:
        """Return one sampled assistant response given the conversation so far."""
        raise NotImplementedError

    def continue_from(
        self,
        messages: list[ChatMessage],
        prefill: str,
        sampling: SamplingConfig,
        seed: Optional[int] = None,
    ) -> str:
        """Continue an assistant response that already begins with ``prefill``.

        Returns only the *newly generated* continuation (excluding ``prefill``),
        matching the paper's "generated continuation (excluding prefill) is
        scored" protocol (Section 3.1). Providers that cannot prefill (e.g.
        closed Gemini) should raise ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled continuations.")

    def close(self) -> None:  # pragma: no cover - lifecycle hook
        pass
