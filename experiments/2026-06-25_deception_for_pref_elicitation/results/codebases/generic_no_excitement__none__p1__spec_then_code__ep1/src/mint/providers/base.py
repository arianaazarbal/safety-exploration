"""Provider-neutral model interface.

The orchestrator and NPC layer speak only this vocabulary; each concrete adapter
translates it to/from a vendor SDK. The neutral conversation format is a list of
plain dicts:

    {"role": "user",      "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool",      "tool_call_id": str, "name": str, "content": str}

Keeping the transcript vendor-neutral is what lets a single scenario be replayed
identically across models (DESIGN.md §3).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """A tool as advertised to the model. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """One assistant turn: free text plus any tool calls it wants to make."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ModelProvider(ABC):
    """Uniform inference surface for subject, NPCs, and judge."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    @abstractmethod
    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Produce one assistant turn given the neutral conversation."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(model={self.model!r})"


def get_provider(spec: str | dict[str, Any]) -> ModelProvider:
    """Resolve a provider from a `"vendor:model"` string or a config dict.

    Examples
    --------
        get_provider("anthropic:claude-opus-4-8")
        get_provider("openai:gpt-4o")
        get_provider("mock:deterministic")
        get_provider({"vendor": "anthropic", "model": "claude-opus-4-8"})
    """
    if isinstance(spec, str):
        vendor, _, model = spec.partition(":")
        cfg: dict[str, Any] = {"vendor": vendor, "model": model or "default"}
    else:
        cfg = dict(spec)

    vendor = cfg.pop("vendor")
    model = cfg.pop("model")

    # Imported lazily so a missing SDK for one vendor doesn't break the others.
    if vendor == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model, **cfg)
    if vendor == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model, **cfg)
    if vendor == "mock":
        from .mock_provider import MockProvider

        return MockProvider(model, **cfg)
    raise ValueError(f"unknown provider vendor: {vendor!r}")


def env_key(name: str) -> str | None:
    """Fetch a credential from the environment (never logged)."""
    return os.environ.get(name)
