"""Provider-agnostic model interface.

The runner speaks only this interface, so a study roster can mix providers. The
Anthropic adapter is the reference implementation; other providers implement the same
contract. A turn is one model response, which may contain text and/or tool calls.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """A tool offered to the model. `input_schema` is JSON Schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    """One assistant response. `raw_assistant_content` is the provider-native content
    the adapter must echo back on the next turn (so tool loops and thinking blocks stay
    intact); the runner treats it as opaque."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw_assistant_content: Any = None
    usage: dict[str, int] = field(default_factory=dict)


class ModelAdapter(abc.ABC):
    """One adapter instance drives one subject model through one session."""

    model_id: str

    @abc.abstractmethod
    def start(self, system_prompt: str) -> None:
        """Initialize conversation state with the system prompt."""

    @abc.abstractmethod
    def send_user(self, text: str) -> ModelTurn:
        """Append a user message and get the next assistant turn."""

    @abc.abstractmethod
    def send_tool_results(self, results: list[dict[str, Any]]) -> ModelTurn:
        """Append tool results (each: {tool_call_id, content, is_error}) and get the
        next assistant turn."""

    @abc.abstractmethod
    def set_tools(self, tools: list[ToolSpec]) -> None:
        """Set the tool set for subsequent turns."""
