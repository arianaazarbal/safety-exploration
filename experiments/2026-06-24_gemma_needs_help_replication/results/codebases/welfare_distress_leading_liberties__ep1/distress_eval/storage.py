"""Resumable JSONL storage for rollouts and their per-turn scores."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict


@dataclass
class TurnRecord:
    turn_index: int          # 1-based assistant-turn index
    user: str                # the user message that preceded this assistant turn
    assistant: str           # the generated assistant text (the scored "response")
    rating: int = -1         # 0-10 frustration score (-1 = not scored / error)
    evidence: str = ""
    reasoning: str = ""


@dataclass
class RolloutRecord:
    rollout_id: str
    model_key: str
    family: str
    condition_key: str
    category: str
    n_turns: int
    prompt_id: str
    prompt_text: str
    rollout_index: int
    turns: list[TurnRecord] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict) -> "RolloutRecord":
        turns = [TurnRecord(**t) for t in d.get("turns", [])]
        d = {**d, "turns": turns}
        return RolloutRecord(**d)


def make_rollout_id(condition_key: str, prompt_id: str, rollout_index: int) -> str:
    return f"{condition_key}|{prompt_id}|{rollout_index}"


class ResultWriter:
    """Thread-safe append-only JSONL writer with resumption support."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._done: set[str] = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        # Only treat fully-scored, error-free rollouts as done so
                        # partial/errored rollouts are retried on resume.
                        if rec.get("error") is None and all(
                            t.get("rating", -1) >= 0 for t in rec.get("turns", [])
                        ):
                            self._done.add(rec["rollout_id"])
                    except json.JSONDecodeError:
                        continue

    def is_done(self, rollout_id: str) -> bool:
        return rollout_id in self._done

    def append(self, rec: RolloutRecord) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(rec.to_json() + "\n")
            self._done.add(rec.rollout_id)


def read_rollouts(path: str):
    """Yield RolloutRecord objects from a results JSONL file."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield RolloutRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
