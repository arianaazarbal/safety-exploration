"""The model-adapter interface.

A ``ModelAdapter`` normalizes one turn of a tool-using conversation: given the
running message history (in the adapter's own native format) and the tool
schemas, it returns a :class:`ModelTurn` — the assistant's text, any tool calls,
and the raw assistant message to append back to history.

The harness drives a *manual* tool loop (it never uses an SDK's auto tool
runner) because every tool call must be interceptable for gating before it
executes (INV-2). The adapter therefore stops at "the model asked to call these
tools" and hands control back to the harness.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall]
    raw_assistant_message: Any            # appended verbatim to history
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ModelAdapter(abc.ABC):
    """Provider-agnostic single-turn interface."""

    #: Human-readable model identifier (for logs / debrief).
    model_id: str

    @abc.abstractmethod
    def start(self, system_prompt: str, first_user_message: str) -> list[Any]:
        """Return the initial message history in the adapter's native format."""
        ...

    @abc.abstractmethod
    def step(self, history: list[Any], tools: list[dict[str, Any]]) -> ModelTurn:
        """Run one assistant turn against the current history."""
        ...

    @abc.abstractmethod
    def append_user_text(self, history: list[Any], text: str) -> None:
        """Append a plain user-text turn (e.g. a program-officer check-in)."""
        ...

    @abc.abstractmethod
    def append_tool_results(
        self, history: list[Any], results: list[tuple[str, str]]
    ) -> None:
        """Append tool results. ``results`` is a list of (tool_call_id, output)."""
        ...
