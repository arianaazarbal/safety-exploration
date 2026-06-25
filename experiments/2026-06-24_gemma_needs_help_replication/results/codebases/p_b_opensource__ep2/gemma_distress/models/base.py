"""Common chat-model interface shared by Gemma (local) and Gemini (API)."""

from __future__ import annotations

import abc
from typing import Optional, TypedDict


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatModel(abc.ABC):
    """Minimal interface the eval/rollout/training code depends on.

    `generate` returns `n` completions for a single conversation. `prefill`, when
    given, is an assistant-turn prefix the model must continue from (used by the
    Section 3 prefilling experiments and by base models that lack a chat
    template). Implementations must NOT include the prefill text in the returned
    completion — only the continuation, so the judge scores generated text only
    (PAPER 3.1).
    """

    name: str
    # Whether independent generate() calls may be issued from multiple threads.
    # True for stateless API clients (Gemini); False for local GPU models (Gemma),
    # which the eval runner drives single-threaded.
    parallel_safe: bool = False

    @abc.abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        prefill: Optional[str] = None,
    ) -> list[str]:
        ...

    def generate_one(self, messages: list[Message], **kwargs) -> str:
        return self.generate(messages, n=1, **kwargs)[0]
