"""The single interface every model backend implements.

Keeping local (Gemma) and API (Gemini, Claude) models behind one interface is
what lets the eval/prefill/training code stay model-agnostic, and is what would
make adding the out-of-scope families (Qwen, OLMo, ...) a config-only change.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenerationResult:
    """A single generation plus light metadata for auditing/cost tracking."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ModelClient(abc.ABC):
    """Abstract model client.

    Implementations MUST honour temperature exactly (the paper fixes it at 1.0)
    and MUST NOT silently truncate inputs. Sampling `n` completions should use
    the backend's native batching where possible.
    """

    def __init__(self, name: str, temperature: float = 1.0, max_new_tokens: int = 2048):
        self.name = name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @abc.abstractmethod
    def chat(
        self,
        messages: list[Message],
        n: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[GenerationResult]:
        """Generate `n` assistant continuations of a chat-formatted conversation."""

    def complete(
        self,
        prompt: str,
        n: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[GenerationResult]:
        """Continue a raw text prompt (used for base-model prefilling).

        Only meaningful for local backends with tokenizer access. API backends
        raise NotImplementedError; the Section 3 prefill study is Gemma-only, so
        this is never required of an API client (see DESIGN.md §Section 3).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support raw-text completion"
        )

    @property
    def supports_prefill(self) -> bool:
        return False
