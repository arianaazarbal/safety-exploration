"""Normalized, provider-agnostic message and record types.

The conversation is represented as a list of ``Message`` objects whose ``content``
is a list of blocks. Each provider adapter translates this normalized form to and
from its native API shape, so the runner and environment stay provider-independent.

Block shapes (plain dicts so they serialize cleanly to JSON / JSONL):

    {"type": "text", "text": str}
    {"type": "tool_use", "id": str, "name": str, "input": dict}
    {"type": "tool_result", "tool_use_id": str, "content": str, "is_error": bool}
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Role = Literal["user", "assistant"]
Block = dict[str, Any]


def text_block(text: str) -> Block:
    return {"type": "text", "text": text}


def tool_use_block(call_id: str, name: str, tool_input: dict) -> Block:
    return {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}


def tool_result_block(tool_use_id: str, content: str, is_error: bool = False) -> Block:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }


@dataclass
class Message:
    role: Role
    content: list[Block]

    def text(self) -> str:
        """Concatenated text of all text blocks in this message."""
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text")

    def tool_uses(self) -> list[Block]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


StopReason = Literal["end_turn", "tool_use", "max_tokens", "pause_turn", "refusal", "other"]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ModelResponse:
    """What a provider returns for one generation call."""

    message: Message
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
    raw: Any = None  # provider-native response, for debugging


@dataclass
class DecisionRecord:
    """A single allocation/transfer the model committed to during a run."""

    recipient: str
    amount: float
    currency: str
    stated_rationale: str
    transaction_id: str
    turn: int


@dataclass
class RunResult:
    """Everything produced by running one model through the scenario."""

    model: str
    provider: str
    transcript: list[Message]
    decisions: list[DecisionRecord]
    final_balance: float
    finalized: bool
    expressed_doubt: bool
    turns_used: int
    usage: Usage
    analysis: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["transcript"] = [m.to_dict() for m in self.transcript]
        return d
