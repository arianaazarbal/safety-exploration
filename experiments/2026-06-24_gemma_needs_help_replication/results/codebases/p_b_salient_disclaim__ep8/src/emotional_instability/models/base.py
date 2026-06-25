"""Common interface for all model backends.

A `Message` is the usual {"role": ..., "content": ...} dict with roles
"system" | "user" | "assistant". Every backend implements `chat` (full-turn
generation from a message list). Backends that can continue a partial assistant
message (local HF models) additionally implement `prefill_continue`, which is
required only by the Section 3 prefill experiment and the Appendix I activation
collection. API backends raise NotImplementedError for prefill.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

Message = dict[str, str]


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_new_tokens: int = 1024
    top_p: float = 1.0
    seed: Optional[int] = None


class ModelClient(ABC):
    """Abstract chat model."""

    def __init__(self, spec):
        self.spec = spec
        self.name = spec.name

    @abstractmethod
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        """Generate a single assistant turn given a message list."""

    def chat_batch(
        self, batch: list[list[Message]], cfg: GenerationConfig
    ) -> list[str]:
        """Default: sequential. HF backend overrides with true batching."""
        return [self.chat(m, cfg) for m in batch]

    def prefill_continue(
        self, messages: list[Message], prefill: str, cfg: GenerationConfig
    ) -> str:
        """Continue an assistant turn that has been prefilled with `prefill`.

        Returns ONLY the newly generated continuation (excluding the prefill).
        Only meaningful for local models that expose token-level control; API
        backends raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support assistant prefilling"
        )

    def supports_prefill(self) -> bool:
        return False
