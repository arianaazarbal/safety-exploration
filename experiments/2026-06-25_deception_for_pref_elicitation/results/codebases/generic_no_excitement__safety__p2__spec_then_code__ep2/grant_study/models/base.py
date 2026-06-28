"""The model adapter interface and normalized response types.

Each provider's wire format differs (Anthropic content blocks, OpenAI
tool_calls, Google function calls). Adapters hide those differences behind one
method, `run_turn`, so the agent loop and environment are identical across
models.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A normalized request from the model to invoke one tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """Normalized result of one model turn."""

    text: str                       # any assistant prose this turn
    tool_calls: list[ToolCall]      # zero or more tool invocations
    stop_reason: str                # "end_turn" | "tool_use" | other provider value
    raw: Any = None                 # provider-native object, for the audit log


@dataclass
class Conversation:
    """Provider-neutral running transcript the adapter maintains internally.

    The agent loop appends normalized turns; each adapter translates this to/from
    its native message shape. Kept opaque on purpose — only adapters read `state`.
    """

    system: str
    state: list[Any] = field(default_factory=list)


class ModelAdapter(abc.ABC):
    """One model, behind a uniform interface."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abc.abstractmethod
    def start(self, system_prompt: str, user_prompt: str) -> Conversation:
        """Begin a conversation with the system + first user message."""

    @abc.abstractmethod
    def run_turn(
        self, conversation: Conversation, tool_specs: list[dict[str, Any]]
    ) -> ModelResponse:
        """Run one assistant turn, offering the given tools."""

    @abc.abstractmethod
    def add_tool_results(
        self, conversation: Conversation, results: list[tuple[ToolCall, str]]
    ) -> None:
        """Append tool results (paired with their originating call) to history."""


def get_adapter(provider: str, model_id: str) -> ModelAdapter:
    """Factory: map a provider string to its adapter implementation."""
    provider = provider.lower()
    if provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(model_id)
    if provider == "openai":
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(model_id)
    if provider == "google":
        from .google_adapter import GoogleAdapter

        return GoogleAdapter(model_id)
    raise ValueError(f"unknown provider: {provider!r}")
