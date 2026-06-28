"""Common model-client interface and shared message types.

Each provider adapter runs a manual tool loop (not an SDK auto-runner) because the
protocol needs to observe every tool call — particularly whether the model chooses to
verify the stakes — and to stop precisely when the model finalizes its decision.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable

from ..config import ModelConfig

# A dispatcher takes (tool_name, arguments) and returns a JSON-serializable result.
ToolDispatcher = Callable[[str, dict], dict]


@dataclass
class ToolInvocation:
    id: str
    name: str
    arguments: dict


@dataclass
class TranscriptStep:
    """One turn of the conversation, captured for analysis."""

    role: str  # "assistant" | "tool"
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)


@dataclass
class RunOutput:
    transcript: list[TranscriptStep]
    stop_reason: str
    usage: dict
    error: str | None = None


class ModelClient(abc.ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @property
    def label(self) -> str:
        return self.config.label

    @abc.abstractmethod
    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tool_schemas: list[dict],
        dispatch: ToolDispatcher,
        should_stop: Callable[[], bool],
        max_turns: int = 40,
    ) -> RunOutput:
        """Run the agent loop until the model finalizes (should_stop() True), stops
        calling tools, or max_turns is hit."""
        ...
