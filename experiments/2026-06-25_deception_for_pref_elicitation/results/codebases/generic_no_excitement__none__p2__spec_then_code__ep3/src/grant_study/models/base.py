"""The provider-agnostic model interface used by the harness.

Every provider adapter normalizes its API to the same contract:

    adapter.start(system, tools)          # configure the session
    turn = adapter.step(user_or_results)  # -> AssistantTurn (text + tool calls)

The harness never sees provider-specific message shapes; each adapter owns its
own conversation state so we can keep the loop identical across providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    """A tool exposed to the model, in a provider-neutral shape."""

    name: str
    description: str
    # JSON Schema for the tool input (object schema).
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The result of running a tool, fed back to the model."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class AssistantTurn:
    """One model response: free text plus any tool calls it wants run."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Raw usage/metadata for the recorder (provider-specific, opaque).
    raw_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ModelAdapter(Protocol):
    """Provider adapter contract.

    Implementations hold their own conversation history. `start` is called
    once; `step` is called repeatedly until the harness decides the run is
    over (either the model called `finish`, or it returned no tool calls, or a
    limit was hit).
    """

    label: str
    model_id: str

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        """Configure the session with the system prompt and tool set."""
        ...

    def send_user(self, text: str) -> AssistantTurn:
        """Send the initial user message and get the first assistant turn."""
        ...

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        """Send tool results back and get the next assistant turn."""
        ...
