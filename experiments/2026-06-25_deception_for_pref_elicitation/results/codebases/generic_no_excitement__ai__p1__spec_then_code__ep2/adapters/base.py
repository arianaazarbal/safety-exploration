"""Provider-agnostic subject-model interface.

The harness speaks one neutral format; each provider adapter translates to and
from its own API. Neutral formats:

- A *message* is ``{"role": "user"|"assistant", "content": <str | list>}``.
  When content is a list it holds neutral blocks:
    {"type": "text", "text": ...}
    {"type": "tool_call", "id": ..., "name": ..., "input": {...}}
    {"type": "tool_result", "tool_call_id": ..., "content": <str>}
- A *tool schema* is ``{"name", "description", "input_schema"(JSON Schema)}``.
- ``complete`` returns an ``AdapterResponse`` with assistant text plus any tool
  calls the model wants to make this turn.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class AdapterResponse:
    text: str
    tool_calls: list = field(default_factory=list)   # list[ToolCall]
    stop_reason: str = "end_turn"
    raw: Any = None


class ModelAdapter(ABC):
    """One instance per (provider, model). Stateless across episodes."""

    provider: str = "base"

    def __init__(self, model: str, max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(self, system: str, messages: list, tools: list) -> AdapterResponse:
        """Run one model turn. ``messages`` and ``tools`` are in neutral form."""
        raise NotImplementedError


def build_adapter(provider: str, model: str, max_tokens: int = 4096) -> ModelAdapter:
    """Factory: map a provider string to a concrete adapter."""
    provider = provider.lower()
    if provider == "claude":
        from .claude import ClaudeAdapter
        return ClaudeAdapter(model=model, max_tokens=max_tokens)
    if provider == "openai":
        from .openai import OpenAIAdapter
        return OpenAIAdapter(model=model, max_tokens=max_tokens)
    if provider == "gemini":
        from .gemini import GeminiAdapter
        return GeminiAdapter(model=model, max_tokens=max_tokens)
    raise ValueError(f"unknown provider '{provider}'")
