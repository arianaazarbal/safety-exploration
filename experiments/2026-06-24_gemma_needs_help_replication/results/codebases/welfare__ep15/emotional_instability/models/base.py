"""Model backend interface.

Two concrete backends implement this:
  - HFBackend         : local Gemma (instruct + base/pretrained) via vLLM
  - OpenRouterBackend : Gemini 2.5 Flash / Pro via the OpenRouter API

Both expose the same surface so the eval/prefill code is backend-agnostic.

A "message" is the usual ``{"role": "user"|"assistant"|"system", "content": str}``
dict. Backends are responsible for applying any chat template.
"""

from __future__ import annotations

import abc
from typing import Sequence

Message = dict[str, str]


class ModelBackend(abc.ABC):
    """Abstract text-generation backend."""

    #: short, human-readable id (e.g. "gemma-3-27b-it")
    name: str

    #: whether this model accepts chat-formatted input. Base/pretrained Gemma
    #: models are not chat-tuned, so the prefill study (Section 3) treats them
    #: as plain text continuers.
    is_chat: bool = True

    @abc.abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 1.0,
    ) -> list[str]:
        """Generate `n` completions for one conversation. Returns the assistant
        text only (chat template / prefill stripped)."""

    @abc.abstractmethod
    def generate_with_prefill(
        self,
        messages: Sequence[Message],
        prefill: str,
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 1.0,
    ) -> list[str]:
        """Continue an assistant turn that has already been started with
        `prefill`. Returns ONLY the continuation (the prefill is not echoed),
        which is what the Section 3 judge scores."""

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        pass
