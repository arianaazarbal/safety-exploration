"""Common chat-model interface shared by the local (Gemma) and API (Gemini)
backends.

The evaluation harness only needs two capabilities:

  1. `generate` a single assistant turn given a chat history (all targets).
  2. `continue_from_prefill` — generate a continuation given a partially
     written final assistant turn (the Section 3 prefill experiment). This is
     only meaningful for open-weights backends, and the closed-API backend
     raises NotImplementedError. The paper notes the same limitation: prefill
     comparisons cannot be run on closed Gemini.
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


@dataclass
class GenerationParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 2048
    stop: list[str] | None = None
    seed: int | None = None


class ChatModel(abc.ABC):
    """Abstract target model.

    Implementations: HFChatModel / VLLMChatModel (open weights, Gemma),
    APIChatModel (closed weights, Gemini).
    """

    def __init__(self, name: str, family: str, role: str):
        self.name = name
        self.family = family
        self.role = role

    @property
    def is_open_weights(self) -> bool:
        return self.role == "full"

    @abc.abstractmethod
    def generate(
        self, messages: list[Message], params: GenerationParams
    ) -> str:
        """Return one assistant turn continuing `messages`."""

    def generate_batch(
        self, conversations: list[list[Message]], params: GenerationParams
    ) -> list[str]:
        """Default batch implementation; backends may override for speed."""
        return [self.generate(c, params) for c in conversations]

    def continue_from_prefill(
        self,
        messages: list[Message],
        prefill: str,
        params: GenerationParams,
    ) -> str:
        """Continue a partially-written final assistant turn.

        `prefill` is the already-written start of the assistant's response;
        the returned string is the *continuation only* (excluding the prefill),
        matching the paper's "generated continuation (excluding prefill) is
        scored" protocol. Open-weights backends only.
        """
        raise NotImplementedError(
            f"continue_from_prefill is not supported for {self.name} "
            "(prefill requires open weights / token-level control)."
        )

    # Tokenizer access is only needed for open-weights flows (prefill
    # truncation, probing). API backends leave these unimplemented.
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        raise NotImplementedError
