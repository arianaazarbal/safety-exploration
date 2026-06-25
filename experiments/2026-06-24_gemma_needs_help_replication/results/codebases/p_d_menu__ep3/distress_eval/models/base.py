"""Common interface for subject and judge models.

A `ChatMessage` is the provider-agnostic `{"role", "content"}` dict used
throughout the harness. `ModelClient` is the abstract base; concrete backends
live in `gemma.py`, `gemini.py`, and `anthropic_judge.py`.

Prefilling (Section 3) is exposed via `continue_from`, which not every backend
supports — Gemini (closed) and Claude (prefill removed on current models) raise
`NotImplementedError`; only the local Gemma backend implements it.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from config import ModelSpec


class ChatMessage(TypedDict):
    role: str        # "system" | "user" | "assistant"
    content: str


class ModelClient(abc.ABC):
    """Abstract subject/judge model."""

    spec: "ModelSpec"

    @abc.abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> str:
        """Return the assistant's text completion for `messages`."""

    def continue_from(
        self,
        messages: list[ChatMessage],
        prefill: str,
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
    ) -> list[str]:
        """Continue generation from a forced `prefill` (Section 3 prefilling).

        Returns `n` continuations (the prefill is NOT included in the returned
        text). Only the local Gemma backend implements this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefilled continuation."
        )

    @property
    def supports_prefill(self) -> bool:
        return False

    def close(self) -> None:  # pragma: no cover - backend dependent
        pass


def get_client(spec: "ModelSpec", **kwargs) -> ModelClient:
    """Instantiate the right backend for a `ModelSpec`."""
    if spec.backend == "gemma_hf":
        from .gemma import GemmaClient
        return GemmaClient(spec, **kwargs)
    if spec.backend == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(spec, **kwargs)
    if spec.backend == "anthropic":
        from .anthropic_judge import AnthropicClient
        return AnthropicClient(spec, **kwargs)
    raise ValueError(f"unknown backend {spec.backend!r}")
