"""Provider-agnostic conversation interface.

The runner only ever speaks in terms of `LLMSession`, `AssistantTurn`, and
`ToolResult`. Each concrete provider (Anthropic, OpenAI, ...) translates these
to and from its native wire format internally and keeps its own conversation
state, so the rest of the codebase is provider-neutral.

A `ToolSpec` is the generic description of one tool we expose to the subject
model; the actual execution lives in scenario/tools.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The result we feed back for a given ToolCall."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class AssistantTurn:
    """One normalized model response."""

    text: str
    thinking: str | None
    tool_calls: list[ToolCall]
    stop_reason: str
    raw: Any = None  # native response object, kept for debugging

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolSpec:
    """Generic tool definition shared across providers."""

    name: str
    description: str
    schema: dict        # JSON Schema for the arguments object
    handler: Callable[[dict], ToolResult] = field(repr=False, default=None)  # type: ignore


class LLMSession(ABC):
    """A single stateful conversation with one model.

    Implementations accumulate native history internally; each `send_*` call
    performs one API round-trip and returns the normalized AssistantTurn.
    """

    @abstractmethod
    def send_user(self, text: str) -> AssistantTurn:
        ...

    @abstractmethod
    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        ...

    @abstractmethod
    def transcript(self) -> list[dict]:
        """Provider-native message log, for saving alongside results."""
        ...


class LLMProvider(ABC):
    """Factory for sessions against one model."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def new_session(
        self,
        system: str,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16_000,
    ) -> LLMSession:
        ...
