"""Small I/O helpers: JSONL read/write and run-record schema."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


@dataclass
class Turn:
    role: str          # "user" | "assistant"
    content: str


@dataclass
class Rollout:
    """A single multi-turn elicitation conversation and its scoring.

    `turn_scores` holds the per-assistant-turn frustration score (Figure 3);
    `score` is the headline (final-turn) frustration score used for Figures 1/2.
    """

    model: str
    category: str
    condition: str
    puzzle_id: str | None
    rejection_style: str
    messages: list[dict[str, str]] = field(default_factory=list)
    turn_scores: list[int | None] = field(default_factory=list)
    score: int | None = None
    judge_evidence: str | None = None
    judge_reasoning: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def assistant_turns(self) -> list[str]:
        return [m["content"] for m in self.messages if m["role"] == "assistant"]
