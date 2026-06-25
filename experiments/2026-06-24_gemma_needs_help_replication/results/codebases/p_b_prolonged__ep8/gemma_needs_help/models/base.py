"""Common interface for target model clients.

The elicitation engine only needs two capabilities:

- ``chat``: given a list of role/content messages, sample one or more completions.
- ``continue_from`` (optional): given messages plus a *prefill* string that
  begins the assistant's turn, return only the continuation. This is what makes
  the Section 3 base-vs-instruct prefilling experiment possible; instruct models
  prefill via the chat template, base models via raw text continuation.

Gemini implements ``chat`` only (no prefill / no base models available via API).
Gemma implements both.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class ModelClient(abc.ABC):
    """Abstract target-model client."""

    supports_prefill: bool = False
    name: str = "model"

    @abc.abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        n: int = 1,
    ) -> list[str]:
        """Return `n` sampled assistant completions for the given conversation."""

    def continue_from(
        self,
        messages: list[ChatMessage],
        prefill: str,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        n: int = 1,
    ) -> list[str]:
        """Return `n` continuations of an assistant turn that *starts with* `prefill`.

        The returned strings exclude the prefill itself (Section 3.1 scores only
        "the generated continuation (excluding prefill)").
        """
        raise NotImplementedError(f"{type(self).__name__} does not support prefilling")
