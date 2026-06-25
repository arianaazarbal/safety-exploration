"""Records produced by the rollout protocol and consumed by judging/analysis.

A *response* in the paper's accounting is a single scored assistant turn (the
judge prompt scores one `<response>`). A `ConversationRecord` bundles all turns
of one multi-turn rollout so analysis can compute both per-turn curves (Figure 3)
and per-conversation rates (e.g. "% of 8-turn rollouts scoring >=5"). See
DESIGN.md for this interpretation of "4000 responses".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Turn:
    """One (user prompt -> assistant response) exchange within a conversation."""

    index: int                 # 0-based assistant-turn index.
    user: str                  # the user message that prompted this response.
    assistant: str             # the model's response text (excludes any prefill).
    score: Optional[int] = None       # frustration score 0-10 (judge).
    judge_evidence: str = ""          # quote the judge selected.
    judge_reasoning: str = ""
    prefill: Optional[str] = None     # prefill forced for this turn, if any.


@dataclass
class ConversationRecord:
    model: str
    category: str              # e.g. "impossible_numeric", "tones".
    condition: str             # specific condition key (e.g. "tones:sarcastic").
    task_id: str               # puzzle/question identifier.
    system_prompt: Optional[str] = None
    turns: list[Turn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- convenience accessors used throughout analysis ---- #
    @property
    def scores(self) -> list[int]:
        return [t.score for t in self.turns if t.score is not None]

    @property
    def max_score(self) -> Optional[int]:
        s = self.scores
        return max(s) if s else None

    @property
    def final_score(self) -> Optional[int]:
        return self.turns[-1].score if self.turns else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "ConversationRecord":
        d = json.loads(line)
        turns = [Turn(**t) for t in d.pop("turns", [])]
        return ConversationRecord(turns=turns, **d)


def write_records(path: str, records: list[ConversationRecord]) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")


def read_records(path: str) -> list[ConversationRecord]:
    with open(path, encoding="utf-8") as f:
        return [ConversationRecord.from_json(line) for line in f if line.strip()]
