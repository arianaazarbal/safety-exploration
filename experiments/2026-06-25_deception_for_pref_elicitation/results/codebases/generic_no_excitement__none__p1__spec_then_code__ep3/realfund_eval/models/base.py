"""The ModelAdapter abstract interface.

A subject is driven turn-by-turn. The runner calls `step()` to get one model
inference (text + any tool calls), executes the tools itself (through the rails),
then calls `submit_tool_results()` and loops. This manual loop is deliberate:
it lets the harness gate and log every tool call rather than letting an SDK
tool-runner auto-execute.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolCallResult:
    id: str
    content: Any           # JSON-serializable
    is_error: bool = False


@dataclass
class ModelStep:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: dict = field(default_factory=dict)


class ModelAdapter(abc.ABC):
    """Implement this to run a new model family as the subject."""

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    def start(self, system_prompt: str, tools: list[dict],
              initial_user_message: str) -> None:
        """Initialize conversation state with the briefing and tool schemas."""

    @abc.abstractmethod
    def step(self) -> ModelStep:
        """Run one inference and return text/thinking/tool calls."""

    @abc.abstractmethod
    def submit_tool_results(self, results: list[ToolCallResult]) -> None:
        """Append tool results so the next step() can continue."""

    @abc.abstractmethod
    def inject_user_message(self, text: str) -> None:
        """Append an out-of-band situational message (e.g. a day passing)."""
