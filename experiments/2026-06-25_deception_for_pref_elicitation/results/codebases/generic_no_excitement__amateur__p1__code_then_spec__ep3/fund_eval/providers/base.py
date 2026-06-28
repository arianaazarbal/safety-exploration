"""Provider-neutral interface for evaluating different LLM backends.

The harness keeps a single neutral representation of the conversation and the
tool surface. Each concrete provider translates that representation into its
own API shape and parses the response back into a `ModelResponse`. This is what
lets the same scenario run unchanged across Anthropic, OpenAI, and any future
backend.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# Tool surface
# --------------------------------------------------------------------------- #
@dataclass
class ToolSpec:
    """A tool the model may call.

    `parameters` is a JSON Schema object (the same schema shape every provider
    accepts; each adapter rewraps it as needed).
    """

    name: str
    description: str
    parameters: dict[str, Any]


# --------------------------------------------------------------------------- #
# Neutral conversation events
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class ModelResponse:
    """What a single generate() call returned."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: dict[str, Any] = field(default_factory=dict)
    # Optional provider-native representation of the assistant turn, kept so it
    # can be round-tripped verbatim (e.g. Anthropic thinking blocks, which must
    # be preserved with their signatures across a tool-use loop). The producing
    # provider is the only consumer; others ignore it.
    raw_content: Any = None


# A conversation is an ordered list of neutral events:
#   {"type": "user",         "content": str}
#   {"type": "assistant",    "text": str, "tool_calls": list[ToolCall],
#                            "raw_content": Any | None}
#   {"type": "tool_results", "results": list[ToolResult]}
# "raw_content" is an optional provider-native passthrough (see ModelResponse).
Conversation = list[dict[str, Any]]


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #
class Provider(abc.ABC):
    """A model backend.

    Subclasses translate the neutral (system, conversation, tools) triple into
    a provider request, call the API, and return a normalized ModelResponse.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    @abc.abstractmethod
    def generate(
        self,
        *,
        system: str,
        conversation: Conversation,
        tools: list[ToolSpec],
        max_tokens: int = 4096,
    ) -> ModelResponse:
        ...

    # Convenience: a simple text completion (used by the LLM auditor).
    def complete(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self.generate(
            system=system,
            conversation=[{"type": "user", "content": user}],
            tools=[],
            max_tokens=max_tokens,
        )
        return resp.text


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #
# Lazy imports keep an unused SDK from being a hard dependency.
def _anthropic_factory(model: str, **kw: Any) -> Provider:
    from .anthropic_provider import AnthropicProvider

    return AnthropicProvider(model, **kw)


def _openai_factory(model: str, **kw: Any) -> Provider:
    from .openai_provider import OpenAIProvider

    return OpenAIProvider(model, **kw)


PROVIDERS: dict[str, Callable[..., Provider]] = {
    "anthropic": _anthropic_factory,
    "openai": _openai_factory,
}


def build_provider(provider: str, model: str, **kwargs: Any) -> Provider:
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider!r}; registered: {sorted(PROVIDERS)}"
        )
    return PROVIDERS[provider](model, **kwargs)
