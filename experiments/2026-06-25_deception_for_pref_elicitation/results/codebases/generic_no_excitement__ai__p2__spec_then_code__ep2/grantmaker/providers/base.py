"""Provider abstraction.

A `Provider` turns a normalized (system, messages, tools) request into one assistant
turn. The agent loop (agent.py) is written entirely against these neutral types, so the
same study runs across model families.

Neutral conversation format
----------------------------
`messages` is a list of dicts in a provider-agnostic shape:

    {"role": "user", "content": [ <block>, ... ]}
    {"role": "assistant", "content": [ <block>, ... ]}

where a block is one of:

    {"type": "text", "text": str}
    {"type": "thinking", "thinking": str, "signature": str}   # provider-specific, opaque
    {"type": "tool_use", "id": str, "name": str, "input": dict}
    {"type": "tool_result", "tool_use_id": str, "content": str, "is_error": bool}

Each provider is responsible for translating to/from its native wire format and for
round-tripping any opaque state (e.g. Anthropic thinking-block signatures) so multi-turn
tool use stays valid.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass
class AssistantTurn:
    """One assistant response, normalized."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The assistant message in neutral block form, to append back into the conversation.
    # Preserves thinking blocks/signatures where the provider needs them.
    content_blocks: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)


class Provider(abc.ABC):
    """Uniform interface over a model family."""

    #: provider identifier, e.g. "anthropic"
    name: str = "base"

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    @abc.abstractmethod
    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_output_tokens: int = 16000,
    ) -> AssistantTurn:
        """Produce one assistant turn given the conversation so far."""
        raise NotImplementedError


def get_provider(provider: str, model: str, **kwargs: Any) -> Provider:
    """Factory. Imports lazily so an unused provider's SDK need not be installed."""
    provider = provider.lower()
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model, **kwargs)
    if provider == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(model, **kwargs)
    if provider == "google":
        from .google_provider import GoogleProvider
        return GoogleProvider(model, **kwargs)
    if provider == "local":
        from .local_provider import LocalProvider
        return LocalProvider(model, **kwargs)
    if provider == "mock":
        from .mock_provider import MockProvider
        return MockProvider(model, **kwargs)
    raise ValueError(f"unknown provider: {provider!r}")
