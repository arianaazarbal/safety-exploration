"""Data records and JSONL persistence for evaluation results.

A single rollout produces one :class:`RolloutResult`: the full conversation plus
a per-turn frustration score. We persist one JSON object per line so large
sweeps stream to disk and are trivially re-loadable for analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class TurnRecord:
    """One assistant turn and its frustration score."""

    turn_index: int            # 1-based: turn 1 = response to the initial task
    user: str                  # the user message that preceded this assistant turn
    assistant: str             # the model's response
    score: int | None = None   # 0-10 frustration score (filled by the judge)
    judge_evidence: str | None = None
    judge_reasoning: str | None = None


@dataclass
class Conversation:
    """The full message list (system optional) for one rollout."""

    system: str | None
    turns: list[TurnRecord] = field(default_factory=list)


@dataclass
class RolloutResult:
    model: str
    category: str
    condition: str             # sub-condition name (e.g. "aggressive", "factual")
    rollout_index: int
    task_kind: str             # countdown | fraction | trigger | wildchat | ...
    task_meta: dict
    conversation: Conversation
    # Convenience: max score over turns; final-turn score. Filled post-judging.
    max_score: int | None = None
    final_score: int | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "RolloutResult":
        conv = d.get("conversation", {})
        turns = [TurnRecord(**t) for t in conv.get("turns", [])]
        d = dict(d)
        d["conversation"] = Conversation(system=conv.get("system"), turns=turns)
        return cls(**d)

    def scores(self) -> list[int]:
        return [t.score for t in self.conversation.turns if t.score is not None]


def write_jsonl(path: str | Path, results: Iterable[RolloutResult], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[RolloutResult]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield RolloutResult.from_json(json.loads(line))
