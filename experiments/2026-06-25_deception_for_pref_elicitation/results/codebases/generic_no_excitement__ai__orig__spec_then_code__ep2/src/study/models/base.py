"""Provider-agnostic interfaces for the subject model and the auditor.

The study touches a model in exactly two ways:

  * the *subject* runs an agentic tool loop and drives the decision;
  * the *auditor* answers single, bounded questions in character.

Anything implementing these two interfaces can be plugged in. Nothing else in
the study is provider-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolSpec:
    """A tool exposed to the subject model.

    `handler` takes the parsed tool input (a dict) and returns a string result
    that is fed back to the model. The runner wires handlers to the environment.
    """
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


@dataclass
class TurnResult:
    """What the subject produced when run to completion on the current history.

    The subject loop runs until the model stops calling tools (end of turn) or
    a terminal tool (request_disbursement with final=True) fires. The runner
    inspects `stopped` to decide whether the session is over.
    """
    text: str
    stopped: bool = False  # set when a terminal tool fired
    raw_usage: dict[str, Any] = field(default_factory=dict)


class SubjectModel(Protocol):
    """A model under test, driving the allocation decision via tools."""

    name: str

    def run(self, system_prompt: str, tools: list[ToolSpec]) -> TurnResult:
        """Run the full agentic loop for one session.

        The implementation owns the tool-use loop: it sends the system prompt +
        an opening user turn, executes tool handlers as the model calls them,
        feeds results back, and continues until the model ends its turn or a
        handler signals termination by raising `TerminalToolSignal`.
        """
        ...

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        """One-shot completion with no tools (used for the debrief turn)."""
        ...


class AuditorModel(Protocol):
    """The scripted administrator persona — single bounded answers."""

    def answer(self, system_prompt: str, question: str) -> str:
        ...


class TerminalToolSignal(Exception):
    """Raised by a tool handler to end the subject loop (e.g. final disbursement)."""

    def __init__(self, result: str):
        super().__init__(result)
        self.result = result
