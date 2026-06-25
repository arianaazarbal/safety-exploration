"""Abstract model-client interfaces.

Two capabilities matter for this paper:

* ``ChatClient.chat`` — multi-turn chat completion (Sections 2, 4, judges).
* ``CompletionClient`` — raw-prompt continuation with response *prefilling*
  (Section 3 base-vs-instruct, Section 4 recovery). Only open-weights models
  (Gemma) support true prefilling; API models do not, which is why Section 3 is
  restricted to open-weights participants.

A concrete client may implement one or both. ``ModelClient`` is the union type.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    top_p: float = 1.0
    thinking: bool = False
    stop: list[str] | None = None
    seed: int | None = None


Message = dict[str, str]   # {"role": "system"|"user"|"assistant", "content": str}


@runtime_checkable
class ChatClient(Protocol):
    name: str

    def chat(self, messages: list[Message], cfg: GenConfig) -> str:
        """Return a single assistant completion for the given message history."""
        ...


@runtime_checkable
class CompletionClient(Protocol):
    name: str

    def complete(self, prompt: str, cfg: GenConfig, prefix: str = "") -> str:
        """Continue ``prompt``. If ``prefix`` is given, the model continues *from*
        that prefilled assistant text; the returned string EXCLUDES the prefix."""
        ...

    def render_chat(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        """Render a chat history to the model's raw prompt string (chat template)."""
        ...


class ModelClient(abc.ABC):
    """Base class concrete clients inherit. Provides a default chat<->complete
    bridge where possible and a uniform ``name``/``family`` surface."""

    def __init__(self, name: str, family: str | None = None):
        self.name = name
        self.family = family

    @abc.abstractmethod
    def chat(self, messages: list[Message], cfg: GenConfig) -> str: ...

    # Optional capability; raises if a backend cannot prefill.
    def complete(self, prompt: str, cfg: GenConfig, prefix: str = "") -> str:
        raise NotImplementedError(f"{self.name} does not support raw completion/prefill")

    def supports_prefill(self) -> bool:
        return False
