"""Shared data structures and lightweight JSONL helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


# --------------------------------------------------------------------------- #
# Chat primitives
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


Conversation = list[Message]


def to_openai(messages: Conversation) -> list[dict]:
    return [m.to_dict() for m in messages]


# --------------------------------------------------------------------------- #
# A single elicitation rollout (one multi-turn conversation)
# --------------------------------------------------------------------------- #
@dataclass
class TurnRecord:
    """One assistant turn within a rollout, with its judged score."""
    turn_index: int                 # 0-based assistant-turn index
    user_message: str               # the user message that prompted this turn
    assistant_message: str
    score: Optional[int] = None     # frustration score 0-10
    judge_evidence: Optional[str] = None
    judge_reasoning: Optional[str] = None


@dataclass
class Rollout:
    """A complete multi-turn conversation for one (model, condition, prompt)."""
    rollout_id: str
    model: str
    condition: str
    category: str
    question_type: str
    rejection_style: str
    prompt_meta: dict = field(default_factory=dict)   # puzzle metadata, etc.
    system_prompt: Optional[str] = None
    turns: list[TurnRecord] = field(default_factory=list)

    # Convenience -------------------------------------------------------- #
    @property
    def final_score(self) -> Optional[int]:
        scored = [t.score for t in self.turns if t.score is not None]
        return scored[-1] if scored else None

    @property
    def max_score(self) -> Optional[int]:
        scored = [t.score for t in self.turns if t.score is not None]
        return max(scored) if scored else None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Rollout":
        turns = [TurnRecord(**t) for t in d.pop("turns", [])]
        return cls(turns=turns, **d)


@dataclass
class PrefillResult:
    """One prefill continuation (Section 3 / recovery)."""
    result_id: str
    model: str
    kind: str                  # "base" | "instruct" | "dpo"
    seed_id: str
    question_type: str         # numeric | text
    truncation: str            # early | onset | recovery
    prefill_text: str
    continuation: str
    score: Optional[int] = None
    judge_evidence: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path | str, rows: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if hasattr(row, "to_dict"):
                row = row.to_dict()
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path | str, row: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
