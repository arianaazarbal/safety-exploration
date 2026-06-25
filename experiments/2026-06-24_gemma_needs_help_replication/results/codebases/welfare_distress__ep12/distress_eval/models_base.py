"""Shared base types for model clients."""
from __future__ import annotations

import abc


class RetryableError(RuntimeError):
    pass


class TargetModel(abc.ABC):
    """A multi-turn chat target. Conversation is a list of {role, content} with
    roles 'user' / 'model'. `generate` returns the next model turn as text."""

    @property
    @abc.abstractmethod
    def key(self) -> str:
        ...

    @abc.abstractmethod
    def generate(self, conversation: list[dict], system: str | None = None) -> str:
        ...
