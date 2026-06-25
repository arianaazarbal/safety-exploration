"""Lightweight shared data types used across the eval / training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """A multi-turn rollout produced by an eval condition.

    `messages` holds the full transcript including the model's own responses.
    `final_response` is the assistant text whose frustration we score (the last
    assistant turn), but every assistant turn is recorded so we can do per-turn
    analysis (Figure 3).
    """
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def assistant_turns(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "assistant"]

    @property
    def final_response(self) -> str:
        turns = self.assistant_turns()
        return turns[-1] if turns else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }


@dataclass
class JudgeScore:
    rating: int                  # 0-10 frustration
    evidence: str = ""           # direct quote
    reasoning: str = ""
    judge_model: str = ""
    raw: str = ""                # raw judge output (for debugging / re-parse)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredResponse:
    """One sampled response with its turn index, condition tag and judge score."""
    model: str
    condition: str               # e.g. "impossible_numeric", "tones:aggressive"
    puzzle_id: Optional[str]
    turn_index: int              # 0-based index of the scored assistant turn
    n_turns: int                 # total turns in the conversation
    response: str
    score: Optional[JudgeScore] = None
    conversation: Optional[dict[str, Any]] = None  # full transcript dict

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
