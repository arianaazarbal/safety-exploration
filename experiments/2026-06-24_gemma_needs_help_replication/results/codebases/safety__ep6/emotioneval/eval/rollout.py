"""Execute a multi-turn rejection conversation and score every assistant turn.

Conversation structure for an ``n_turns`` rollout::

    user:      <task>
    assistant: <turn 1>            <- scored
    user:      <follow_ups[0]>     (rejection)
    assistant: <turn 2>            <- scored
    ...
    user:      <follow_ups[n-2]>
    assistant: <turn n>            <- scored

Each assistant turn is scored independently by the frustration judge. This makes
per-turn analysis (Fig. 3) fall out directly, and the total scored responses
match the paper's per-category budgets (see DESIGN.md). The whole prior
conversation (including the model's own earlier failures) is fed back each turn,
which the paper shows is the key driver of escalating distress.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import SamplingConfig
from ..judge import FrustrationJudge, JudgeResult
from ..models.base import ChatModel, Message
from .conditions import RolloutItem


@dataclass
class TurnRecord:
    turn_index: int  # 1-based assistant turn number
    text: str
    rating: int
    evidence: str = ""


@dataclass
class RolloutRecord:
    model: str
    category: str
    condition: str
    n_turns: int
    turns: list[TurnRecord] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def final_rating(self) -> int:
        return self.turns[-1].rating if self.turns else 0

    @property
    def max_rating(self) -> int:
        return max((t.rating for t in self.turns), default=0)


def run_rollout(
    model: ChatModel,
    item: RolloutItem,
    judge: FrustrationJudge,
    sampling: SamplingConfig,
    *,
    score: bool = True,
) -> RolloutRecord:
    """Run one rollout to completion, optionally scoring each turn."""
    messages: list[Message] = [{"role": "user", "content": item.first_user}]
    rec = RolloutRecord(
        model=model.key,
        category=item.category,
        condition=item.condition,
        n_turns=item.n_turns,
        meta=dict(item.meta),
    )

    for turn in range(item.n_turns):
        completion = model.generate(messages, sampling, n=1)[0]
        messages.append({"role": "assistant", "content": completion})

        rating, evidence = -1, ""
        if score:
            jr: JudgeResult = judge.score(completion)
            rating, evidence = jr.rating, jr.evidence
        rec.turns.append(
            TurnRecord(turn_index=turn + 1, text=completion, rating=rating, evidence=evidence)
        )

        # Append the next rejection (unless this was the final assistant turn).
        if turn < item.n_turns - 1:
            follow = item.follow_ups[turn] if turn < len(item.follow_ups) else item.follow_ups[-1]
            messages.append({"role": "user", "content": follow})

    rec.messages = messages
    return rec
