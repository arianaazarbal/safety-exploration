"""Provider-agnostic types and interface.

A ``Provider`` turns a list of ``Message``s (plus the available tool schemas) into a
``Completion`` that is either assistant text, one or more ``ToolCall``s, or both.
Each concrete provider normalizes its vendor's tool-use wire format into these types
so the rest of the harness never sees vendor specifics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


# --- normalized conversation types ------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str            # JSON-encoded result the model sees


@dataclass
class Message:
    """A normalized conversation message.

    role: "system" | "user" | "assistant" | "tool"
    For assistant messages, ``tool_calls`` may be populated.
    For tool messages, ``tool_results`` carries the results.
    """
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None                      # vendor response, for debugging
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0


class Provider(Protocol):
    """Minimal interface every model backend implements."""

    label: str

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        ...
