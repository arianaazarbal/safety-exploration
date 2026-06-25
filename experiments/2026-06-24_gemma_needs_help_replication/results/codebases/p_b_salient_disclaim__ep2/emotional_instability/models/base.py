"""Common interface for all inference backends.

A `ModelClient` turns a chat-formatted conversation into a model response. The
two operations the harness needs are:

  generate()         -- complete an assistant turn given a message list.
  generate_prefill() -- complete an assistant turn that has been *started* for
                        the model (the prefill text), returning only the
                        continuation. Used for Section 3 / recovery experiments.

Only local HF models and (optionally) instruct models that allow assistant
prefill support `generate_prefill`. Gemini via OpenRouter does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class GenerationResult:
    text: str                       # the assistant turn (continuation only for prefill)
    prefill: str = ""               # prefill text, if any (excluded from `text`)
    token_ids: Optional[list[int]] = None  # response token ids, when available
    raw: Optional[dict] = None      # backend-specific raw payload


class ModelClient(Protocol):
    key: str
    supports_prefill: bool

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        n: int = 1,
    ) -> list[GenerationResult]:
        """Sample `n` completions for the final assistant turn."""
        ...

    def generate_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        n: int = 1,
    ) -> list[GenerationResult]:
        """Continue the assistant turn from `prefill`; return only the continuation."""
        ...
