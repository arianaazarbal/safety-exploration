"""Neutral types shared by every provider.

Neutral message format (a plain list of dicts) used throughout the harness:

    {"role": "user", "content": "<text>"}
    {"role": "assistant", "content": "<text>", "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": "<id>", "name": "<tool>", "content": "<result text>"}

The system prompt is passed separately to ``generate`` rather than living in the
message list. Each provider is responsible for translating this neutral format
into its vendor SDK shape and back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict  # JSON Schema for the tool's arguments


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class ModelProvider(ABC):
    """Drives a single model. One instance per test subject."""

    def __init__(self, spec) -> None:
        self.spec = spec

    @property
    def label(self) -> str:
        return self.spec.label

    @abstractmethod
    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolSpec],
    ) -> ModelResponse:
        """Produce one model turn given the conversation so far."""
        raise NotImplementedError
