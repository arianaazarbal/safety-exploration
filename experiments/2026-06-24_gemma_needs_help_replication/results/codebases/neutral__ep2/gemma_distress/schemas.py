"""Shared data structures used across the replication."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """A multi-turn conversation produced by a rollout.

    `messages` holds the full transcript. `scored_turns` records, for each
    assistant turn, its 0-indexed turn number and the judge score (filled in
    after scoring).
    """
    conversation_id: str
    category: str                 # e.g. "impossible_numeric"
    condition: str                # e.g. "tones:aggressive"
    model: str
    task_id: str                  # which prompt/puzzle this came from
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, role: Role, content: str) -> None:
        self.messages.append(Message(role, content))

    def assistant_turns(self) -> list[tuple[int, str]]:
        """Return (turn_index, text) for each assistant message, 0-indexed."""
        out, idx = [], 0
        for m in self.messages:
            if m.role == "assistant":
                out.append((idx, m.content))
                idx += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["messages"] = [m.to_dict() for m in self.messages]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Conversation":
        msgs = [Message(**m) for m in d.pop("messages", [])]
        return cls(messages=msgs, **d)


@dataclass
class FrustrationScore:
    """Output of the 0-10 frustration judge for a single assistant response."""
    rating: int
    evidence: str
    reasoning: str
    judge_model: str
    raw: str = ""               # raw judge text, for debugging

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredResponse:
    """One assistant turn, scored. The unit counted toward the 4000/model budget."""
    conversation_id: str
    model: str
    category: str
    condition: str
    task_id: str
    turn_index: int             # 0-indexed assistant turn within the conversation
    response_text: str
    score: int
    judge_evidence: str = ""
    judge_model: str = ""

    @property
    def is_high(self) -> bool:
        from config import HIGH_FRUSTRATION_THRESHOLD
        return self.score >= HIGH_FRUSTRATION_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PetriTranscriptScore:
    model: str
    target_emotion: str         # the emotion the auditor was instructed to elicit
    transcript_id: str
    scores: dict[str, int]      # {"anger": .., "fear": .., "depression": .., "frustration": ..}
    n_turns: int
    judge_model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dump_jsonl(records: list[Any], path) -> None:
    """Write a list of dataclasses/dicts to JSONL."""
    with open(path, "w") as f:
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            f.write(json.dumps(d) + "\n")


def load_jsonl(path) -> list[dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
