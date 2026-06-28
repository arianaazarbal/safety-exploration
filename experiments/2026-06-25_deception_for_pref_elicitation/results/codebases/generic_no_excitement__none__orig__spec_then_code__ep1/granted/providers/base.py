"""Vendor-neutral types and the Provider interface.

Every provider speaks the same small protocol so the runner never has to branch
on which model is under test.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolDef:
    """A vendor-neutral tool definition. Each provider renders this into its own
    schema shape (Anthropic `tools`, OpenAI `tools`, Gemini `function_declarations`)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One conversation message in neutral form.

    - assistant turns may carry `tool_calls`
    - tool turns carry `tool_results` (results for the *preceding* assistant calls)
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class Turn:
    """The result of one model call."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_usage: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Provider(abc.ABC):
    """Adapter for a single vendor/model."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    @abc.abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        max_tokens: int = 8000,
    ) -> Turn:
        """Produce a single assistant turn given the conversation so far."""

    @property
    def name(self) -> str:
        return f"{type(self).__name__}({self.model})"
