"""Provider-agnostic model interface.

The runner speaks a normalized transcript + tool format; each adapter translates
to/from its provider's wire format. This keeps `runner.py` provider-agnostic so the
same agentic loop drives Claude, GPT, and Gemini identically.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """A tool offered to the model, in JSON-Schema-parameter form."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object for the arguments


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    """A model turn in the normalized transcript."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Opaque provider-native content, stored so adapters can echo it back
    # verbatim on the next call (preserves thinking blocks / signatures).
    provider_raw: Any = None
    provider: str | None = None


@dataclass
class ToolResultTurn:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class UserTurn:
    content: str


# A transcript is an ordered list of these turns.
Turn = UserTurn | AssistantTurn | ToolResultTurn


@dataclass
class GenerateResult:
    turn: AssistantTurn
    stop_reason: str
    usage: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """One instance per model under test."""

    provider: str = "base"

    def __init__(self, model_id: str, max_tokens: int = 8000, effort: str = "high"):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.effort = effort

    @abstractmethod
    def generate(
        self,
        system: str,
        transcript: list[Turn],
        tools: list[ToolSpec],
    ) -> GenerateResult:
        """Produce the next assistant turn given the conversation so far."""

    # Convenience for adapters that don't carry raw content.
    @staticmethod
    def _args_to_json(args: dict[str, Any]) -> str:
        return json.dumps(args, sort_keys=True)
