"""Abstract model-client interface.

All providers are normalized to a single `complete()` call over a list of
provider-agnostic `Message`s, optionally exposing `ToolSpec`s. The return value
is a `ModelResponse` carrying the text, any tool calls the model requested, and
token usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    # For role == "tool": the id of the tool call this result answers.
    tool_call_id: str | None = None
    # For role == "assistant": any tool calls the model emitted (provider-agnostic).
    tool_calls: list["ToolCall"] = field(default_factory=list)


@dataclass
class ToolSpec:
    name: str
    description: str
    # JSON Schema for the tool's input.
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None  # provider-native response object, for debugging/audit

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ModelClient(ABC):
    """A provider-agnostic chat client."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Run one completion. Implementations translate to/from the provider."""
        raise NotImplementedError
