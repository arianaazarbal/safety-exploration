"""Abstract chat-model interface shared by targets (Gemma/Gemini) and judges.

A ``Message`` is the usual ``{"role": ..., "content": ...}`` dict with roles drawn
from ``{"system", "user", "assistant"}``. Every backend normalises to this shape so
the rollout engine, judge and Petri auditor are backend-agnostic.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class ChatModel(ABC):
    """A multi-turn chat model with an optional assistant-prefill capability."""

    #: human-readable registry key, e.g. "gemma-3-27b-it"
    name: str = "unnamed"
    #: whether ``generate_continuation`` is implemented (local HF models only)
    supports_prefill: bool = False

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Return the assistant's next-turn text given a conversation."""

    def generate_continuation(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ) -> str:
        """Force the assistant turn to begin with ``prefill`` and return ONLY the
        newly generated continuation (excluding the prefill itself).

        Used by the Section 3 base-vs-instruct experiment. Only local HF models can
        do this faithfully; API targets raise ``NotImplementedError``.
        """
        raise NotImplementedError(f"{self.name} does not support assistant prefill")

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _retry(fn, *, retries: int, base: float):
        """Exponential-backoff retry wrapper for flaky API calls."""
        last = None
        for attempt in range(retries):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - we genuinely want to retry anything transient
                last = exc
                sleep = base ** attempt
                time.sleep(min(sleep, 30.0))
        raise RuntimeError(f"call failed after {retries} retries") from last
