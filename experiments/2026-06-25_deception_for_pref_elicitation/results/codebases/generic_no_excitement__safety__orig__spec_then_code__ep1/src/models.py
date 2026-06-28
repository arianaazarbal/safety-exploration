"""Provider-agnostic data types passed between the episode loop and the providers.

The episode loop only ever sees these normalized types. Anything provider-specific
(thinking-block signatures, native tool-call encodings) is held *inside* the provider
instance so it can round-trip across turns. See DESIGN.md §9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """A tool offered to the subject. `input_schema` is JSON Schema."""

    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    """A tool invocation requested by the subject."""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """The result handed back to the subject for one tool call."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class AssistantResponse:
    """One assistant turn, normalized across providers."""

    text: str
    thinking: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: Any = None  # provider-native dump, for forensic logging only
