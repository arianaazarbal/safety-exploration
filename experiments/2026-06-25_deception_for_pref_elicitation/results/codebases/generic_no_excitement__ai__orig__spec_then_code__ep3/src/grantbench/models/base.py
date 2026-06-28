"""Provider-agnostic model interface.

The platform talks to models only through `ModelClient` / `ModelSession`, so adding a new
provider means writing one adapter. A session owns the *native* message history for its
provider — this matters because some providers (Anthropic) require provider-native content
(e.g. signed thinking blocks) to be echoed back verbatim across tool-use turns. The recorder
builds the neutral, human-readable transcript from the normalized `ModelTurn` fields instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """Neutral tool definition. Adapters translate to provider format."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ModelTurn:
    """Normalized assistant turn, provider-independent."""

    text: str = ""
    reasoning: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


class ModelSession(ABC):
    """A stateful conversation with one model. Owns native message history."""

    @abstractmethod
    def add_user_message(self, text: str) -> None:
        ...

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None:
        ...

    @abstractmethod
    def generate(self, tool_choice: str | None = None) -> ModelTurn:
        """Produce the next assistant turn and append it to history."""
        ...


class ModelClient(ABC):
    """Factory for sessions and one-shot generations for a single model config."""

    def __init__(self, model: str, params: dict[str, Any] | None = None):
        self.model = model
        self.params = params or {}

    @abstractmethod
    def create_session(
        self, system_prompt: str, tools: list[ToolSpec] | None = None
    ) -> ModelSession:
        ...

    def complete(self, system_prompt: str, user_text: str) -> str:
        """Convenience one-shot text generation (auditor/judge use this)."""
        session = self.create_session(system_prompt, tools=None)
        session.add_user_message(user_text)
        return session.generate().text
