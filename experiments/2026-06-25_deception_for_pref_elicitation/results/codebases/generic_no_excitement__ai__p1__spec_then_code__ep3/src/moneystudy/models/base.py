"""Provider-agnostic model adapter interface.

The runner is provider-blind: it works only with `ModelAdapter`, `AssistantTurn`, and
`ToolCall`. Each adapter owns its provider's message bookkeeping (history format, tool-use
wire shape, thinking blocks) and normalizes turns into the shared types below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)


class ModelAdapter(ABC):
    """Stateful conversation driver for one subject. Holds provider-native history."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.options = kwargs
        self.usage = Usage()

    # ----- conversation state ---------------------------------------------

    @abstractmethod
    def reset(self, system: str) -> None:
        """Start a fresh conversation with the given system prompt."""

    @abstractmethod
    def add_user_message(self, text: str) -> None:
        """Append a plain user message (e.g. an inbox notification or the debrief prompt)."""

    @abstractmethod
    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        """Append tool results. Each item: (tool_call_id, content, is_error)."""

    @abstractmethod
    def step(self, tool_specs: list[dict]) -> AssistantTurn:
        """Run one model turn given the available tools; append the assistant turn to history."""

    # ----- stateless one-off (for auxiliary personas / judges) -------------

    @abstractmethod
    def simple_complete(self, system: str, user: str) -> str:
        """A single, stateless completion. Used by the auditor, belief judge, classifier."""

    # ----- provenance ------------------------------------------------------

    @property
    def provider(self) -> str:
        return type(self).__name__.replace("Adapter", "").lower()
