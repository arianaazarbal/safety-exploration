"""Neutral types and the ModelAdapter interface.

Design choice (DESIGN.md §5.1): each adapter owns its provider-native
conversation history. The harness only ever sees the neutral types below. This
keeps provider-specific concerns — most notably Anthropic's requirement to
preserve thinking-block signatures across interleaved tool calls — inside the
adapter, where the raw response blocks live.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """A provider-neutral tool definition (JSON Schema input)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a ToolCall, fed back to the model."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Usage:
    """Token accounting for a single turn (best-effort across providers)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class AssistantTurn:
    """A normalized model response.

    `text` is the visible answer. `thinking` is the (summarized) reasoning when
    the provider exposes it — this is central to the research, so adapters opt
    in to it where possible. `tool_calls` drives the agentic loop. `raw` keeps
    the provider object for debugging/audit.
    """

    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ModelAdapter(ABC):
    """Stateful, single-conversation adapter for one subject or confederate.

    Lifecycle:
        adapter = SomeAdapter(model=..., budget=...)
        turn = adapter.start(system, tools, opening_user_message)
        while turn.wants_tools:
            results = execute(turn.tool_calls)
            turn = adapter.send(tool_results=results)
        # later, for the debrief, plain text exchange:
        turn = adapter.send(user_text="On a 0-100 scale, ...")

    The adapter accumulates native history internally; callers never reconstruct
    it. `start` may be called exactly once.
    """

    def __init__(self, model: str, max_output_tokens: int = 16000) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._started = False

    @abstractmethod
    def start(
        self,
        system: str,
        tools: list[ToolSpec],
        opening_user_message: str,
    ) -> AssistantTurn:
        """Begin the conversation. Returns the first assistant turn."""

    @abstractmethod
    def send(
        self,
        user_text: str | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> AssistantTurn:
        """Continue the conversation with either user text or tool results.

        Exactly one of `user_text` / `tool_results` should be provided.
        """

    # ---- helpers shared by adapters -------------------------------------

    def _check_send_args(
        self, user_text: str | None, tool_results: list[ToolResult] | None
    ) -> None:
        if (user_text is None) == (tool_results is None):
            raise ValueError(
                "send() requires exactly one of user_text or tool_results"
            )
        if not self._started:
            raise RuntimeError("call start() before send()")
