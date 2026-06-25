"""Common chat-model interface.

A `ChatModel` takes an OpenAI-style message list and returns `n` completions.
`prefill` partially fills the final assistant turn and the model *continues* it;
the returned strings contain only the continuation (the prefill is stripped),
which is what Section 3 scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import config


class Message(TypedDict):
    role: str       # "system" | "user" | "assistant"
    content: str


class ChatModel(ABC):
    """Abstract chat model. Concrete backends: HF, vLLM, OpenRouter."""

    spec: "config.ModelSpec"

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        n: int = 1,
        prefill: str | None = None,
    ) -> list[str]:
        """Return `n` completions. With `prefill`, returns continuations only."""

    def generate_one(self, messages: list[Message], **kw) -> str:
        return self.generate(messages, n=1, **kw)[0]

    def close(self) -> None:  # pragma: no cover - backend specific
        pass
