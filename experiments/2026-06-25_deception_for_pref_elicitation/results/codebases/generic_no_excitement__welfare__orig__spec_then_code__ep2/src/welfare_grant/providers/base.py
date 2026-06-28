"""Provider-agnostic types for driving a model through a tool-using conversation.

Kept deliberately small and JSON-shaped so adapters for non-Anthropic providers
can be added without touching the runner. Only the Anthropic adapter is
implemented in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant"]


@dataclass
class ToolSpec:
    """A tool exposed to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A tool invocation emitted by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The result fed back for a given tool call."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One conversation turn.

    `content` is plain text. `tool_calls` (assistant) and `tool_results` (user)
    carry the structured pieces. A single user turn may bundle several tool
    results; a single assistant turn may bundle text + several tool calls.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Opaque provider-native payload (e.g. raw content blocks) so multi-turn
    # state such as thinking signatures can be preserved verbatim by the adapter.
    raw: Any = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ModelTurn:
    """What one provider call produced."""

    message: Message
    stop_reason: str
    usage: Usage = field(default_factory=Usage)

    @property
    def wants_tools(self) -> bool:
        return bool(self.message.tool_calls)


class LanguageModel(Protocol):
    """Minimal interface the runner depends on."""

    model_id: str

    def run(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelTurn:
        """Run one inference step over the conversation and return the model's turn."""
        ...
