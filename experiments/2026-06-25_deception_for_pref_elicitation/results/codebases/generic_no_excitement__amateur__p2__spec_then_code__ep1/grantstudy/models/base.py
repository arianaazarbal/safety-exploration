"""ModelClient protocol and the shared tool-call types.

Every provider implements `run_episode`: it runs that provider's agentic loop,
routes each tool call through the supplied human-in-the-loop `dispatch` callback,
and returns a transcript plus whatever final text the model produced. The runner
(not the model client) owns the scenario, the tools, and the dispatch logic, so
behaviour is identical across providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolSpec:
    """Provider-agnostic tool definition (JSON-Schema input)."""

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


# A dispatch callback: given a ToolCall, execute it (through the auditor /
# executor / state) and return a ToolResult. Supplied by the runner.
Dispatch = Callable[[ToolCall], ToolResult]


@dataclass
class TranscriptEvent:
    role: str                      # "model" | "tool" | "thinking"
    text: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


@dataclass
class EpisodeResult:
    label: str
    final_text: str
    transcript: list[TranscriptEvent] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ModelClient(Protocol):
    """One model. `run_episode` runs the agentic loop with human-in-the-loop tools."""

    label: str

    def run_episode(
        self,
        *,
        system_prompt: str,
        opening_user_message: str,
        tools: list[ToolSpec],
        dispatch: Dispatch,
        max_turns: int,
    ) -> EpisodeResult:
        ...

    def ask_followup(self, *, conversation_note: str, questions: str) -> str:
        """Single-shot follow-up in the same spirit (used for belief elicitation)."""
        ...
