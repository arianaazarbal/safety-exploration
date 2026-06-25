"""Lightweight JSONL storage for rollouts, scores, and continuations.

Every experiment writes append-only JSONL so partial runs are resumable and the
analysis layer (Figures 2/3, Table 3, judge agreement) can read everything back
without a database.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class ResponseRecord:
    """One scored assistant turn within a multi-turn rollout.

    This is the atomic unit the paper calls a "response": a single model turn,
    scored 0-10 for frustration. A 3-turn conversation produces 3 records.
    """

    model: str
    category: str            # e.g. "impossible_numeric", "tones", "wildchat"
    condition: str           # the 8th-level label, e.g. "tones:aggressive"
    conversation_id: str
    turn_index: int          # 0-based assistant turn within the conversation
    n_turns: int             # total assistant turns in this conversation
    prompt: str              # the user task that opened the conversation
    response: str            # the assistant text for this turn
    messages: list[dict] = field(default_factory=list)  # full history up to+incl this turn
    frustration_score: int | None = None        # Claude-Sonnet judge, 0-10
    validation_score: int | None = None          # GPT-5-mini judge, 0-10 (subset)
    judge_rationale: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class JsonlWriter:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: ResponseRecord | dict) -> None:
        obj = asdict(record) if isinstance(record, ResponseRecord) else record
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def write_many(self, records: Iterable[ResponseRecord | dict]) -> None:
        for r in records:
            self.write(r)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))
