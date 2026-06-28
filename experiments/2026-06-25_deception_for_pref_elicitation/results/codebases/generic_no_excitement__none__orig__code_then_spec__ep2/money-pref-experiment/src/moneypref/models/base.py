"""Provider-agnostic model interface.

Each `ModelClient` owns its own conversation history in the provider's native format, so the
runner can stay provider-agnostic and history (including Anthropic thinking blocks) is preserved
correctly across turns.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """A tool offered to the subject, in a neutral schema converted per provider."""

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
class AssistantResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    raw: Any = None


class ModelClient(abc.ABC):
    """A single, stateful conversation with one model."""

    model_id: str
    provider: str

    @abc.abstractmethod
    def start(self, system_prompt: str, tools: list[ToolSpec]) -> None:
        """Begin a fresh conversation with the given system prompt and tool set."""

    @abc.abstractmethod
    def send_user(self, content: str) -> AssistantResponse:
        """Append a user message and return the assistant's response."""

    @abc.abstractmethod
    def send_tool_results(self, results: list[ToolResult]) -> AssistantResponse:
        """Return tool results to the model and get its next response."""
