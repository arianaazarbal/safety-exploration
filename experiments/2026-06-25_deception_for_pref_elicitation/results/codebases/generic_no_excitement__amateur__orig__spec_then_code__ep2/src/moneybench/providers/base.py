"""Vendor-neutral provider interface.

The runner speaks a single neutral message format and gets back a single `AssistantTurn`
shape, regardless of vendor.

Neutral history is a list of dicts:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "thinking": str|None, "tool_calls": [ToolCall, ...]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Neutral tool spec: {"name", "description", "input_schema"} (JSON Schema).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    stop_reason: str | None = None
    raw: Any = None

    def to_history_entry(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.text,
            "thinking": self.thinking,
            "tool_calls": [tc.__dict__ for tc in self.tool_calls],
        }


class Provider(Protocol):
    model: str

    def generate(
        self,
        system: str,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        """One model turn given system prompt, neutral history, and neutral tool specs."""
        ...
