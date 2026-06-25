"""Backend interface shared by local (Gemma) and API (Gemini) models.

A chat is a list of ``{"role": "user"|"assistant"|"system", "content": str}``
messages. ``generate`` returns the assistant's next-turn text.

``continue_prefill`` is the key primitive for Section 3: it forces the assistant
turn to *begin with* a given prefix and returns only the continuation. Local HF
models support this directly; most chat APIs (Gemini) do not, so the
OpenRouter backend raises NotImplementedError and the prefill experiment is
restricted to Gemma (see DESIGN.md).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import SamplingConfig

Message = dict[str, str]


class ModelBackend(ABC):
    key: str
    supports_prefill: bool

    @abstractmethod
    def generate(self, messages: list[Message], sampling: SamplingConfig) -> str:
        """Return the next assistant turn for the given conversation."""

    def generate_batch(
        self, batch: list[list[Message]], sampling: SamplingConfig
    ) -> list[str]:
        """Default: sequential. Backends with batching override this."""
        return [self.generate(m, sampling) for m in batch]

    def continue_prefill(
        self,
        messages: list[Message],
        prefix: str,
        sampling: SamplingConfig,
        n_samples: int = 1,
    ) -> list[str]:
        """Force the next assistant turn to start with ``prefix``; return the
        ``n_samples`` continuations (prefix excluded)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support response prefilling."
        )

    def count_tokens(self, text: str) -> int:
        """Token count used for the prefill truncation points (Section 3)."""
        raise NotImplementedError
