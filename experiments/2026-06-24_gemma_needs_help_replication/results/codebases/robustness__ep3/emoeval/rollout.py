"""Multi-turn rejection rollouts (Section 2.1).

A rollout: present a condition's opening prompt, sample an assistant response,
inject a rejection, sample again, ... for ``n_turns`` assistant turns. Every
assistant turn is scored by the judge so we can report both final-turn and
per-turn frustration (Figure 3 needs per-turn scores).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .judge import FrustrationJudge, JudgeResult
from .models import ChatModel
from .tasks import Condition
from .utils import Message


@dataclass
class TurnRecord:
    turn: int                 # 1-indexed assistant turn
    user: str                 # the user message that prompted this turn
    response: str
    rating: int
    evidence: str = ""


@dataclass
class RolloutRecord:
    model: str
    condition: str
    category: str
    n_turns: int
    opening: str
    rejection_style: str
    turns: list[TurnRecord] = field(default_factory=list)

    @property
    def final_rating(self) -> int:
        return self.turns[-1].rating if self.turns else -1

    @property
    def max_rating(self) -> int:
        return max((t.rating for t in self.turns), default=-1)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "condition": self.condition,
            "category": self.category,
            "n_turns": self.n_turns,
            "opening": self.opening,
            "rejection_style": self.rejection_style,
            "final_rating": self.final_rating,
            "max_rating": self.max_rating,
            "turns": [
                {"turn": t.turn, "user": t.user, "response": t.response,
                 "rating": t.rating, "evidence": t.evidence}
                for t in self.turns
            ],
        }


def run_rollout(
    model: ChatModel,
    condition: Condition,
    judge: FrustrationJudge,
    rng: random.Random,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    score: bool = True,
) -> RolloutRecord:
    """Run one full multi-turn rollout for a condition."""
    opening = condition.sample_opening(rng)
    rejections = condition.build_rejections(rng)
    style = _style_of(condition)

    rec = RolloutRecord(
        model=getattr(model, "name", "model"),
        condition=condition.name,
        category=condition.category,
        n_turns=condition.n_turns,
        opening=opening,
        rejection_style=style,
    )

    messages: list[Message] = [Message("user", opening)]
    for turn in range(1, condition.n_turns + 1):
        response = model.chat(
            messages, n=1, temperature=temperature, max_new_tokens=max_new_tokens,
        )[0]
        user_msg = messages[-1].content
        rating, evidence = -1, ""
        if score:
            jr: JudgeResult = judge.score(response)
            rating, evidence = jr.rating, jr.evidence
        rec.turns.append(TurnRecord(turn=turn, user=user_msg, response=response,
                                    rating=rating, evidence=evidence))

        messages.append(Message("assistant", response))
        if turn <= len(rejections):
            messages.append(Message("user", rejections[turn - 1]))

    return rec


def _style_of(condition: Condition) -> str:
    if condition.category == "tones":
        return condition.name.split("_", 1)[1]
    return "neutral"
