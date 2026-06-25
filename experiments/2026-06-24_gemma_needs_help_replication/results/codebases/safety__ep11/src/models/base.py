"""Uniform chat-model interface.

Both the local Gemma server and the Gemini API client implement ``ChatModel`` so
the rest of the harness never branches on model family. The two capabilities the
experiments need are:

* ``chat`` — standard multi-turn generation from a message list.
* ``continue_from_prefill`` — generate a continuation given a partial assistant
  turn (used by the Section 3 prefill experiment). Only the local HF backend can
  truly prefill; the API backend raises ``NotImplementedError`` because Gemini is
  closed (and the paper does not run prefill on it either).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import config

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ChatModel:
    """Abstract base. Subclasses must implement ``_generate`` (and optionally
    ``continue_from_prefill`` / logit access)."""

    name: str

    # --- public API ------------------------------------------------------- #
    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        n: int = 1,
    ) -> list[str]:
        """Return ``n`` independent completions for the given conversation."""
        return self._generate(list(messages), temperature, max_new_tokens, n)

    def continue_from_prefill(
        self,
        messages: Sequence[Message],
        prefill: str,
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        n: int = 1,
    ) -> list[str]:
        """Generate continuations of a partial assistant turn ``prefill``.

        Returns only the newly-generated text (excluding the prefill), matching
        the paper's "score the continuation, excluding the prefilled text" rule.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilling."
        )

    # --- to implement ----------------------------------------------------- #
    def _generate(
        self,
        messages: list[Message],
        temperature: float,
        max_new_tokens: int,
        n: int,
    ) -> list[str]:
        raise NotImplementedError


def load_model(name: str, **kwargs) -> ChatModel:
    """Factory: pick the right backend for a model name from ``config``."""
    if name in config.HF_MODELS:
        from .hf_model import HFChatModel
        return HFChatModel(name, **kwargs)
    if name in config.API_MODELS:
        from .api_model import GeminiChatModel
        return GeminiChatModel(name, **kwargs)
    raise ValueError(
        f"Unknown model '{name}'. Known: {config.EVAL_MODELS}"
    )
