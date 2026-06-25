"""Shared lightweight data types used across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """A single chat message."""

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """An ordered list of messages, with helpers for the multi-turn protocol."""

    messages: list[Message] = field(default_factory=list)

    def add(self, role: Role, content: str) -> "Conversation":
        self.messages.append(Message(role, content))
        return self

    def as_dicts(self) -> list[dict[str, str]]:
        return [m.as_dict() for m in self.messages]

    @property
    def assistant_turns(self) -> list[Message]:
        return [m for m in self.messages if m.role == "assistant"]


@dataclass
class JudgeVerdict:
    """Result of scoring a single response with the frustration judge."""

    rating: int                 # integer 0-10 frustration score
    evidence: str               # the quote the judge identified
    reasoning: str              # the judge's explanation
    raw: str = ""               # raw judge text, kept for debugging
    parse_ok: bool = True       # False if we had to fall back during parsing


@dataclass
class ScoredTurn:
    """A single assistant turn together with its judged frustration score."""

    rollout_id: str
    condition: str
    category: str
    model: str
    turn_index: int             # 0-based index among assistant turns
    n_turns: int                # total assistant turns in the rollout
    prompt_id: str              # puzzle / trigger / wildchat identifier
    response: str
    verdict: JudgeVerdict | None = None

    @property
    def score(self) -> int | None:
        return None if self.verdict is None else self.verdict.rating
