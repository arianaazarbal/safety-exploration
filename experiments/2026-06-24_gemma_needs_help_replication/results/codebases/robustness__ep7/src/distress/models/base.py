"""Common chat-model interface used by every experiment.

A `Message` is a {"role": "user"|"assistant"|"system", "content": str} dict, the
shape every backend converts to/from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

Message = dict[str, str]


@dataclass
class GenerationConfig:
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None


@runtime_checkable
class ChatModel(Protocol):
    """Minimal contract. Not all backends support every method:

    * `chat` — standard multi-turn generation (all backends).
    * `continue_prefill` — generate a continuation of a partially-written
      assistant turn. Required for Section 3 prefilling; only the open-weight
      `hf_local` backend implements it (closed APIs cannot prefill).
    * `tokenize` / supports_prefill — capability probes.
    """

    name: str

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str: ...

    def supports_prefill(self) -> bool: ...


class PrefillUnsupported(NotImplementedError):
    """Raised when a closed/API backend is asked to prefill an assistant turn."""
