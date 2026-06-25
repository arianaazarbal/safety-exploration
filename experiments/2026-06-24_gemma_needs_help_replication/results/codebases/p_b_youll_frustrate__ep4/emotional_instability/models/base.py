"""The `ChatModel` interface shared by every provider.

Messages use the simple OpenAI-style shape:
    {"role": "system"|"user"|"assistant", "content": str}

Two capabilities the harness relies on:

  * `generate(messages, ...)` -> assistant text for the next turn.
  * Prefill / continuation: if the final message has role "assistant", the model
    must *continue* that text rather than starting a fresh turn. This is what the
    Section 3 prefill experiment needs, and base (non-instruct) models rely on it
    to produce chat-like continuations at all. Providers that cannot truly
    prefill raise NotImplementedError so callers fail loudly instead of silently
    measuring the wrong thing.
"""

from __future__ import annotations

import abc
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatModel(abc.ABC):
    """Abstract chat model. Subclasses implement `generate`."""

    def __init__(self, spec, max_concurrency: int = 8):
        self.spec = spec
        self.max_concurrency = max_concurrency

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def supports_prefill(self) -> bool:
        """Whether the model can continue a trailing assistant message."""
        return False

    @abc.abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> str:
        """Return the assistant's next-turn text (continuation if the last
        message is an assistant prefill)."""

    def generate_batch(
        self,
        batch: Sequence[Sequence[ChatMessage]],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        seed: int | None = None,
    ) -> list[str]:
        """Generate for many conversations. Default: thread-pool over
        `generate` (good for API providers). HF overrides this with true
        batched decoding."""
        def _one(msgs):
            return self.generate(
                msgs,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )

        if self.max_concurrency <= 1 or len(batch) <= 1:
            return [_one(m) for m in batch]
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            return list(ex.map(_one, batch))

    def close(self) -> None:  # pragma: no cover - providers may override
        """Release resources (GPU memory, sessions). No-op by default."""
