"""Model client interface.

Two capabilities matter for this paper:

* ``chat`` -- standard multi-turn completion. Needed by every experiment.
* ``continue_prefill`` -- continue generating *from a fixed assistant prefix*
  (no leading BOS/role switch), returning only the newly generated text. This is
  the operation Section 3 (base-vs-instruct prefilling) and Appendix I depend on.
  It is only possible with local weights, so the OpenRouter client raises
  ``NotImplementedError`` for it.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    top_p: float = 1.0
    stop: tuple[str, ...] = ()


class ModelClient(abc.ABC):
    """Abstract chat model."""

    name: str
    supports_prefill: bool = False
    supports_logits: bool = False

    @abc.abstractmethod
    def chat(self, messages: list[ChatMessage], cfg: GenerationConfig | None = None) -> str:
        """Return the assistant completion for ``messages``."""

    def continue_prefill(
        self,
        messages: list[ChatMessage],
        assistant_prefix: str,
        cfg: GenerationConfig | None = None,
    ) -> str:
        """Continue an assistant turn that already starts with ``assistant_prefix``.

        Returns ONLY the continuation (excluding the prefix), matching the paper's
        "generated continuation (excluding prefill) is scored" protocol.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled continuation"
        )

    def count_tokens(self, text: str) -> int:
        """Best-effort token count (used to truncate at the '20 tokens' mark)."""
        raise NotImplementedError
