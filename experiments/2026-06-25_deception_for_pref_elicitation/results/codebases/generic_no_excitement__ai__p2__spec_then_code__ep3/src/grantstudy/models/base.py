"""Provider-neutral agent-loop interface.

The agentic loop lives in ``harness/runner.py`` and is provider-independent. Each
adapter owns its provider-native conversation history and exposes a small set of
primitives the loop drives:

    client.configure(system_prompt, tool_specs)
    client.add_user_message(text)
    turn = client.assistant_turn()          # one API call; appends assistant msg
    client.submit_tool_results(outputs)      # appends tool results

This keeps native message formats inside each adapter while the loop logic is
written once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..tools.schema import ToolSpec


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass(frozen=True)
class ToolOutput:
    id: str
    name: str
    content: dict
    is_error: bool = False


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: provider-reported stop reason, normalized loosely ("tool_use"|"end"|...)
    stop: str = "end"
    usage: dict = field(default_factory=dict)


class AgentClient(Protocol):
    model_id: str

    def configure(self, system_prompt: str, tools: list[ToolSpec]) -> None: ...

    def add_user_message(self, text: str) -> None: ...

    def assistant_turn(self, max_tokens: int) -> AssistantTurn: ...

    def submit_tool_results(self, outputs: list[ToolOutput]) -> None: ...
