"""Backend-agnostic chat interface.

Every backend (local HF Gemma, OpenRouter Gemini, Anthropic, OpenAI) implements
``ChatClient``. The two capabilities the experiments need are:

  * ``chat(messages, ...)``           -> generate an assistant reply to a
                                         multi-turn conversation.
  * ``continue_prefill(messages, prefill, ...)`` -> generate a continuation of a
                                         partially-written assistant turn (used
                                         by the Section 3 prefill experiment and
                                         the Section 4 recovery experiment).

Base (pretrained) models additionally support raw text completion; that is
handled inside the HF backend, which knows whether it is a base or instruct
checkpoint.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


class ChatClient(abc.ABC):
    """Abstract chat/generation client."""

    supports_prefill: bool = False
    is_base: bool = False

    @abc.abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        seed: int | None = None,
    ) -> list[str]:
        """Return ``n`` assistant completions for the conversation."""

    def continue_prefill(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        seed: int | None = None,
    ) -> list[str]:
        """Continue a partially written final assistant turn.

        Returns the *continuation only* (excludes the prefill), matching the
        paper's scoring convention ("the generated continuation, excluding
        prefill, is scored").
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefill/continuation."
        )

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        pass
