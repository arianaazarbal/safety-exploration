"""Backend abstraction shared by local (vLLM) and API (OpenRouter / Anthropic)
model wrappers.

A conversation is a list of ``Message`` dicts ``{"role": ..., "content": ...}``
with roles in {"system", "user", "assistant"}. The key method is ``chat_batch``,
which generates one assistant continuation per conversation. ``prefill`` lets us
force the assistant turn to *start* with given text (used for Section 3); backends
that cannot do this (closed API models) raise ``PrefillNotSupported``.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


class PrefillNotSupported(RuntimeError):
    pass


@dataclass
class GenConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    n: int = 1                     # samples per prompt
    seed: int | None = None
    stop: list[str] | None = None


class ModelBackend(abc.ABC):
    """Generates assistant continuations for batches of conversations."""

    def __init__(self, spec):
        self.spec = spec
        self.name = spec.name

    @abc.abstractmethod
    def chat_batch(
        self,
        conversations: list[list[Message]],
        cfg: GenConfig,
        prefill: list[str] | None = None,
    ) -> list[list[str]]:
        """Return, for each conversation, a list of ``cfg.n`` completions.

        If ``prefill`` is given (one string per conversation), each completion is
        the model's continuation *after* that prefilled assistant text; the
        returned string EXCLUDES the prefill (paper scores continuations only).
        """

    def chat(self, conversation: list[Message], cfg: GenConfig) -> str:
        return self.chat_batch([conversation], cfg)[0][0]

    @abc.abstractmethod
    def complete_batch(self, prompts: list[str], cfg: GenConfig) -> list[list[str]]:
        """Raw text completion (no chat template). Used for base-model prefill."""

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        pass
