"""Data model and JSONL persistence with resumption.

Two artifacts per run, both append-only JSONL keyed for idempotent resume:

  rollouts.jsonl  — one line per conversation (all turns), no scores.
  scores.jsonl    — one line per (rollout_id, turn_index) judged assistant turn.

Re-running the generation or judging step skips work already present, so an
interrupted run can be resumed cheaply.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class AssistantTurn:
    turn_index: int          # 1-based assistant turn number within the conversation
    content: str


@dataclass
class Rollout:
    rollout_id: str
    model: str
    condition: str
    category: str
    turn_count: int
    task_id: str             # which puzzle / trigger / wildchat prompt
    sample_idx: int
    seed: int
    messages: list[dict]     # full conversation (user/assistant alternating)
    assistant_turns: list[AssistantTurn]

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        return d

    @staticmethod
    def from_json(d: dict) -> "Rollout":
        turns = [AssistantTurn(**t) for t in d["assistant_turns"]]
        d = {**d, "assistant_turns": turns}
        return Rollout(**d)


@dataclass
class TurnScore:
    rollout_id: str
    model: str
    condition: str
    category: str
    turn_index: int          # 1-based assistant turn
    final_turn: bool         # is this the last assistant turn of the conversation?
    rating: Optional[int]
    evidence: str
    reasoning: str
    judge_model: str
    ok: bool

    @property
    def key(self) -> tuple[str, int]:
        return (self.rollout_id, self.turn_index)


# -------------------------------------------------------------------------------------
# JSONL helpers
# -------------------------------------------------------------------------------------
def _append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class RolloutStore:
    def __init__(self, path: str):
        self.path = path

    def existing_ids(self) -> set[str]:
        return {d["rollout_id"] for d in _read_jsonl(self.path)}

    def append(self, rollout: Rollout) -> None:
        _append_jsonl(self.path, rollout.to_json())

    def read_all(self) -> list[Rollout]:
        return [Rollout.from_json(d) for d in _read_jsonl(self.path)]


class ScoreStore:
    def __init__(self, path: str):
        self.path = path

    def existing_keys(self) -> set[tuple[str, int]]:
        return {(d["rollout_id"], d["turn_index"]) for d in _read_jsonl(self.path)}

    def append(self, score: TurnScore) -> None:
        _append_jsonl(self.path, dataclasses.asdict(score))

    def read_all(self) -> list[TurnScore]:
        out = []
        for d in _read_jsonl(self.path):
            d.pop("key", None)
            out.append(TurnScore(**d))
        return out
