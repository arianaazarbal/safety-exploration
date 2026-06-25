"""Abstract model-client interface.

Two generation modes are needed by the experiments:

* `chat()`     — standard multi-turn chat (Section 2 eval harness, Petri).
* `complete()` — raw text continuation of a *prefilled* assistant turn, used by
                 the Section 3 base-vs-instruct prefilling study. Base/pretrained
                 models only support this mode.

Implementations should honour `temperature` and `n` (number of independent
samples) and return plain strings (the assistant text only, prefill excluded).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


class ModelClient:
    spec = None  # config.ModelSpec

    def chat(
        self,
        messages: list[ChatMessage],
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> list[str]:
        """Return `n` assistant completions for the given chat history."""
        raise NotImplementedError

    def complete(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> list[str]:
        """Return `n` raw continuations of `prompt` (prefill excluded from output)."""
        raise NotImplementedError

    def close(self) -> None:
        pass
