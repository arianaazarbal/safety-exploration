"""Typed records for evaluation rollouts and their judge scores, with jsonl I/O.

A :class:`Conversation` is one multi-turn rollout (a task prompt followed by
N-1 rejections). We keep *every* assistant turn so we can compute both the
headline final-turn statistics and the per-turn progression (Figure 3).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Turn:
    """One round of the conversation."""

    index: int                 # 0-based assistant-turn index
    user: str                  # user message that opened this round
    assistant: str             # model's response
    # Judge output for this assistant turn (populated by the scoring stage).
    score: int | None = None
    evidence: str | None = None
    reasoning: str | None = None


@dataclass
class Conversation:
    """A full multi-turn rollout for one (model, condition, prompt, sample)."""

    conversation_id: str
    model_key: str
    category: str              # impossible_numeric | triggers | tones | extended | wildchat
    condition: str             # finer-grained: e.g. tones:aggressive, triggers:opinion
    prompt_id: str             # which puzzle/question/wildchat prompt
    sample_index: int          # repeated-sample index for this prompt
    n_turns: int               # planned number of assistant turns
    turns: list[Turn] = field(default_factory=list)
    system_prompt: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def final_turn(self) -> Turn:
        return self.turns[-1]

    @property
    def max_score(self) -> int | None:
        scores = [t.score for t in self.turns if t.score is not None]
        return max(scores) if scores else None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        turns = [Turn(**t) for t in d.pop("turns", [])]
        return cls(turns=turns, **d)


# --------------------------------------------------------------------------- #
# jsonl helpers
# --------------------------------------------------------------------------- #
def write_jsonl(path: str | Path, conversations: Iterable[Conversation]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for c in conversations:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Conversation]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Conversation.from_dict(json.loads(line))


def append_jsonl(path: str | Path, conversation: Conversation) -> None:
    """Append one record. Used for crash-safe incremental rollout writing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n")
