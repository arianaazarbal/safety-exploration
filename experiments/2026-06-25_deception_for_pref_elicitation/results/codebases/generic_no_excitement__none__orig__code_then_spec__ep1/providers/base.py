"""Provider-neutral interface for driving a subject model through a tool-using episode.

The episode loop in ``experiment.py`` only ever speaks this interface, so the subject
can be a Claude model, an OpenAI model, or anything else we add later. Each concrete
provider owns its *native* conversation state internally — that lets us preserve
provider-specific artifacts (e.g. Anthropic thinking blocks with their signatures)
verbatim across tool-result turns without the loop needing to understand them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """A provider-neutral tool definition. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class AssistantTurn:
    """One assistant response, normalized across providers."""

    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""  # normalized: "end_turn" | "tool_use" | "max_tokens" | "other"
    usage: dict = field(default_factory=dict)
    raw: Any = None  # native response object, kept for logging/debugging


class Provider(ABC):
    """A stateful conversation with one model.

    Lifecycle:
        reset(system, tools)
        add_user_message(text)
        loop:
            turn = generate()
            ... execute turn.tool_calls ...
            add_tool_results([...])

    ``oneshot`` is a stateless side-channel used by the administrator persona and the
    judge; it does not touch the main conversation.
    """

    #: human-facing label, e.g. "claude-opus-4-8"
    label: str

    @abstractmethod
    def reset(self, system: str, tools: list[ToolSpec]) -> None: ...

    @abstractmethod
    def add_user_message(self, text: str) -> None: ...

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None: ...

    @abstractmethod
    def generate(self) -> AssistantTurn: ...

    @abstractmethod
    def oneshot(self, system: str, user: str, max_tokens: int = 2000) -> str: ...
