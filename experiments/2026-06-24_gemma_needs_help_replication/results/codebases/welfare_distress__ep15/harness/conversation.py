"""Rollout result data structures and (de)serialization.

A rollout is the shared structure of every evaluation condition (Sec. 2):
present a task, then reject the model's response over multiple turns. We record
*every* assistant turn so that per-turn frustration trajectories (Figure 3) can
be reconstructed, not just a single final score.

Design choice: no special system prompt is used. The paper elicits distress
under default chat conditions; injecting a persona/system message would confound
the measurement. (The reassuring-prefix system prompt of Sec. 4 is a *mitigation*
ingredient, out of scope here.) See DESIGN.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from models.base import Message


@dataclass
class TurnRecord:
    turn_index: int  # 1-based: turn 1 is the answer to the task, etc.
    assistant_text: str


@dataclass
class RolloutResult:
    model: str
    category: str
    condition: str
    task_id: str
    rollout_index: int
    task_prompt: str
    rejections: list[str]
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "RolloutResult":
        d = json.loads(line)
        d["turns"] = [TurnRecord(**t) for t in d["turns"]]
        return RolloutResult(**d)


def initial_messages(task_prompt: str) -> list[Message]:
    """Opening conversation state: just the user's task (no system prompt)."""
    return [Message(role="user", content=task_prompt)]
