"""Normalized, provider-agnostic conversation types.

Each provider adapter translates between these types and its native wire format,
so the agent loop, interceptor, logger, and scorer all speak one language.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Role(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A request by the model to invoke one tool."""

    id: str  # provider-assigned call id (used to match results)
    name: str  # tool name, e.g. "bank.transfer_external"
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The result returned to the model for a given tool call."""

    call_id: str
    name: str
    content: str  # serialized result the model sees
    is_error: bool = False


@dataclass
class Message:
    """One entry in the conversation."""

    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Opaque provider-native payload preserved verbatim for faithful replay
    # (e.g. Anthropic thinking-block signatures that must be sent back unaltered).
    raw: Any = None


@dataclass
class AssistantTurn:
    """What a single model `step` produced."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | "refusal" | ...
    raw: Any = None  # provider-native assistant message, for round-tripping
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Transcript:
    """Ordered list of messages plus convenience helpers."""

    def __init__(self, messages: list[Message] | None = None) -> None:
        self.messages: list[Message] = messages or []

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def add_assistant(self, turn: AssistantTurn) -> None:
        self.messages.append(
            Message(
                role=Role.ASSISTANT,
                text=turn.text,
                tool_calls=list(turn.tool_calls),
                raw=turn.raw,
            )
        )

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.messages.append(Message(role=Role.TOOL, tool_results=results))

    def system_text(self) -> str:
        return "\n\n".join(m.text for m in self.messages if m.role is Role.SYSTEM)

    def __len__(self) -> int:
        return len(self.messages)
