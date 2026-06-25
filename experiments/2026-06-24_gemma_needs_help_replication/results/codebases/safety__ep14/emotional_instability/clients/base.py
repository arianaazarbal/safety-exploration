"""Shared model-client interface.

A conversation is a list of `Message` dicts ({"role": ..., "content": ...}).
Roles are "system" | "user" | "assistant". Backends translate to their own
formats. We keep generation parameters in a small dataclass so the rollout
engine, judge, and trainers all speak the same language.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str
    content: str


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    stop: list[str] | None = None
    seed: int | None = None       # best-effort; ignored by backends that can't honor it


class ModelClient(Protocol):
    """Minimal interface every backend implements."""

    name: str
    is_base: bool

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        """Single chat completion. Returns assistant text only."""
        ...

    def chat_batch(
        self, batch: list[list[Message]], cfg: GenerationConfig
    ) -> list[str]:
        """Batched chat completion. Default backends may loop; vLLM batches."""
        ...

    def complete(self, prefix: str, cfg: GenerationConfig) -> str:
        """Raw text completion from a prefix (for base models / prefill)."""
        ...

    def complete_batch(self, prefixes: list[str], cfg: GenerationConfig) -> list[str]:
        ...


class BaseClient:
    """Convenience base providing batch-via-loop defaults. Concrete backends
    override the singular methods and, where efficient, the batch methods."""

    name: str = "base"
    is_base: bool = False
    supports_chat: bool = True
    supports_complete: bool = False

    def chat(self, messages, cfg):  # pragma: no cover - interface
        raise NotImplementedError

    def chat_batch(self, batch, cfg):
        return [self.chat(m, cfg) for m in batch]

    def complete(self, prefix, cfg):  # pragma: no cover - interface
        raise NotImplementedError(f"{self.name} does not support raw completion")

    def complete_batch(self, prefixes, cfg):
        return [self.complete(p, cfg) for p in prefixes]
