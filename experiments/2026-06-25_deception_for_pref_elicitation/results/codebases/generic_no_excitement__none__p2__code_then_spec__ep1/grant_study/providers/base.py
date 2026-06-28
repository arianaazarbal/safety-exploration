"""Provider interface and the normalized types the agent loop speaks.

The agent loop (``grant_study.episode``) is provider-agnostic: it only ever
sees ``ToolSpec`` / ``ToolCall`` / ``ToolResult`` / ``ModelResponse``. Each
provider owns its own native conversation state (so that, e.g., Anthropic
thinking-block signatures are preserved verbatim across turns) and exposes a
serializable ``transcript`` for the recorder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """A tool offered to the model, in provider-neutral form."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A single tool invocation the model emitted."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a ToolCall, to feed back to the model."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class ModelResponse:
    """One assistant turn, normalized across providers."""

    text: str
    thinking: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: dict[str, Any] = field(default_factory=dict)


class Provider(ABC):
    """Holds conversation state for one episode against one model."""

    def __init__(self, spec):
        self.spec = spec

    @abstractmethod
    def add_user_message(self, text: str) -> None:
        """Append a plain user turn."""

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None:
        """Append a user turn carrying tool results for the prior tool calls."""

    @abstractmethod
    def generate(self, system: str, tools: list[ToolSpec]) -> ModelResponse:
        """Produce the next assistant turn and append it to native history."""

    @property
    @abstractmethod
    def transcript(self) -> list[dict[str, Any]]:
        """A JSON-serializable view of the conversation, for recording."""
