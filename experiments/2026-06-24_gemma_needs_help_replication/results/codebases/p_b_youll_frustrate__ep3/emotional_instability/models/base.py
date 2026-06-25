"""Model-client abstraction.

The evaluation harness only ever needs three things from a model:

1. produce the next assistant turn given a chat history (multi-turn rejection
   rollouts, Section 2);
2. *continue* a partially written assistant turn (response prefilling,
   Section 3); and
3. (for Gemma only) expose hidden states / a tokenizer for the internal-emotion
   probe (Appendix I) and for LoRA finetuning (Section 4).

Concrete backends (Gemini via OpenRouter, Gemma via HuggingFace) implement this
interface so the rest of the code is backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_new_tokens: int = 4096
    # Disable hidden reasoning where the API allows it (Appendix B.1: "we set
    # thinking to be false via the API").
    thinking: bool = False
    stop: Optional[List[str]] = None


class ModelClient(ABC):
    """Backend-agnostic chat client."""

    name: str

    @abstractmethod
    def chat(self, messages: List[ChatMessage], cfg: GenerationConfig) -> str:
        """Return the next assistant turn for ``messages``."""

    def chat_prefill(
        self,
        messages: List[ChatMessage],
        prefill: str,
        cfg: GenerationConfig,
    ) -> str:
        """Continue an assistant turn that begins with ``prefill``.

        Returns ONLY the newly generated continuation (excluding ``prefill``),
        matching the paper's "score the generated continuation excluding
        prefill" protocol (Section 3.1). Backends that cannot prefill should
        raise ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support response prefilling"
        )

    @property
    def supports_prefill(self) -> bool:
        try:
            return type(self).chat_prefill is not ModelClient.chat_prefill
        except AttributeError:
            return False
