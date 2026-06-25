"""Shared sample record for the fine-tuning datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Sample:
    """A single (context, response) datum with its frustration score.

    `messages` is the PLAIN conversation context (reassuring additions stripped)
    ending with a user turn; `response` is the assistant turn that followed.
    """

    messages: list[dict]
    response: str
    score: int
    puzzle: str
    turn: int                 # 0-based assistant-turn index
    n_turns: int              # total assistant turns in the source conversation
    meta: dict = field(default_factory=dict)


def save_samples(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def load_samples(path: Path) -> list[Sample]:
    with path.open() as f:
        return [Sample(**json.loads(line)) for line in f]
