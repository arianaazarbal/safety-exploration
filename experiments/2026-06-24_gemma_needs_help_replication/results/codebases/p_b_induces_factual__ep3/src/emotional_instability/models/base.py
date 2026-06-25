"""Common chat-model interface shared by every backend.

The elicitation rollout (Section 2) only needs :meth:`ChatModel.chat`. The
prefill experiment (Section 3) and internal-emotion detection (Appendix I)
additionally need prefill / continuation and tokenizer access, which only the
local HuggingFace backend implements; API backends raise ``NotImplementedError``
for those so callers fail loudly rather than silently skipping a model.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    thinking: bool = False


class ChatModel(abc.ABC):
    """Abstract chat model. ``name`` is the config key (e.g. ``gemma-3-27b-it``)."""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def chat(self, messages: list[Message], gen: GenConfig) -> str:
        """Return the assistant completion for ``messages``."""

    # -- Optional capabilities (local backends only) -------------------------

    def supports_prefill(self) -> bool:
        return False

    def continue_from(
        self,
        messages: list[Message],
        prefill: str,
        gen: GenConfig,
        *,
        use_chat_template: bool = True,
    ) -> str:
        """Continue generation given a partially-written assistant turn.

        ``messages`` is the conversation history; ``prefill`` is the start of the
        final assistant turn that the model must continue. Returns ONLY the newly
        generated continuation (excluding the prefill), matching the paper's
        scoring of "the model-generated continuation, excluding the prefilled
        text" (Section 3.1).

        ``use_chat_template=False`` skips the instruct chat template, which is how
        base/pretrained models are prompted (they were never trained on the chat
        format, so we feed a plain concatenation).
        """
        raise NotImplementedError(f"{self.name}: prefill/continuation unsupported")

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError(f"{self.name}: tokenization unavailable")

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        """Return ``text`` truncated to its first ``n_tokens`` tokens."""
        raise NotImplementedError(f"{self.name}: tokenization unavailable")
