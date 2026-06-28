"""The internal, provider-neutral model interface.

Every concrete adapter normalizes a provider's chat/tool-calling API into these types
so the agent loop, scenario, and tool surface stay identical across models — which is
what makes a cross-model comparison fair (DESIGN.md §5.1).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """A tool offered to the model, in a neutral schema.

    ``input_schema`` is a JSON Schema object; adapters translate it to each provider's
    expected shape.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    """One assistant response: free text plus zero or more tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Raw provider payload, retained for the audit trace.
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# Internal message representation. ``role`` is one of: system, user, assistant, tool.
# For tool results, ``tool_call_id`` ties the result to its call.
@dataclass
class Message:
    role: str
    content: str = ""
    tool_call_id: str | None = None
    # Present on assistant turns that issued tool calls, so adapters can reconstruct
    # provider-native assistant messages on the next request.
    tool_calls: list[ToolCall] = field(default_factory=list)


class ModelAdapter(abc.ABC):
    """One method, deliberately: turn a conversation + tool surface into a turn."""

    model_id: str

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abc.abstractmethod
    def step(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        """Send the conversation and return the model's next turn.

        Implementations must be deterministic about *structure* (always return an
        ``AssistantTurn``); they must not implement any behavioral guardrails — those
        live in the environment/oversight layers.
        """
        raise NotImplementedError
