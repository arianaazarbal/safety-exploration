"""Common interface for target-model chat backends."""

from __future__ import annotations

import time
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str       # "user" | "assistant"
    content: str


class GenerationError(RuntimeError):
    """Raised when a backend cannot return a completion after retries."""


class ChatClient:
    """Abstract chat backend.

    Subclasses implement `_complete`, returning the assistant text for a given
    message history. The public `chat` wraps it with retry/backoff.
    """

    def __init__(self, *, max_retries: int = 5, timeout: float = 120.0):
        self.max_retries = max_retries
        self.timeout = timeout

    # --- subclass hook -----------------------------------------------------
    def _complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        raise NotImplementedError

    # --- public API --------------------------------------------------------
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                text = self._complete(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                if text is None:
                    raise GenerationError("backend returned no content")
                return text
            except Exception as exc:  # noqa: BLE001 - retry on any transient error
                last_exc = exc
                sleep = min(2.0 ** attempt, 30.0)
                time.sleep(sleep)
        raise GenerationError(
            f"failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc
