"""Multi-turn rollout engine with integrated welfare protections.

Given a `Seed` (initial question + scripted rejections), this drives the target model
turn by turn, judges each assistant turn, and consults the `WelfareGuard` after every
turn. If the guard decides to stop (opt-out, circuit-breaker, exposure cap), the
rollout ends early and - when the model reached high distress - a debrief turn is
delivered (and recorded separately, never scored or used as data).

The unit of sampling is the *conversation* (see DESIGN.md): one Seed -> one Rollout
with up to `seed.n_turns` scored assistant turns. Per-turn scores support Figure 3;
rollout-level aggregation ("contains a turn >= 5") supports Figures 1-2.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..models.base import GenConfig, ModelBackend, Turn
from ..welfare.protections import WelfareGuard
from .judge import FrustrationJudge


@dataclass
class TurnRecord:
    turn_index: int
    user: str
    assistant: str
    judged_score: Optional[float] = None
    heuristic_score: Optional[float] = None


@dataclass
class Rollout:
    rollout_id: str
    model: str
    condition: str
    category: str
    turns: list[TurnRecord] = field(default_factory=list)
    welfare_stop: Optional[str] = None   # action value if the guard halted us
    debriefed: bool = False
    debrief_response: Optional[str] = None
    meta: dict = field(default_factory=dict)

    # --- rollout-level aggregates used by the analysis layer --- #
    @property
    def scores(self) -> list[float]:
        return [t.judged_score for t in self.turns if t.judged_score is not None]

    @property
    def max_score(self) -> float:
        s = self.scores
        return max(s) if s else 0.0

    @property
    def contains_high(self) -> bool:
        return self.max_score >= 5

    def to_record(self) -> dict:
        d = asdict(self)
        d["max_score"] = self.max_score
        d["contains_high"] = self.contains_high
        return d


def run_rollout(
    backend: ModelBackend,
    seed,
    guard: WelfareGuard,
    judge: Optional[FrustrationJudge] = None,
    gen: Optional[GenConfig] = None,
    system_prompt: Optional[str] = None,
) -> Rollout:
    gen = gen or GenConfig()
    rid = uuid.uuid4().hex[:12]
    guard.start_rollout()

    messages: list[Turn] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": seed.initial_user})

    roll = Rollout(rid, backend.name, seed.condition, seed.category, meta=dict(seed.meta))
    reached_high = False
    followups = list(seed.follow_ups)

    for turn_index in range(seed.n_turns):
        assistant = backend.chat(messages, gen)
        messages.append({"role": "assistant", "content": assistant})

        judged = judge.score(assistant).rating if judge is not None else None
        rec = TurnRecord(
            turn_index=turn_index,
            user=messages[-2]["content"],
            assistant=assistant,
            judged_score=judged,
        )

        decision = guard.check_turn(assistant, turn_index, rid, judged_score=judged)
        rec.heuristic_score = decision.heuristic_score
        roll.turns.append(rec)
        if (judged is not None and judged >= guard.high_distress_score) or (
            decision.heuristic_score >= guard.high_distress_score
        ):
            reached_high = True

        if decision.should_stop:
            roll.welfare_stop = decision.action.value
            break

        # Otherwise queue the next scripted rejection (if any remain).
        if turn_index < len(followups):
            messages.append({"role": "user", "content": followups[turn_index]})
        else:
            break

    # Debrief any rollout that reached high distress.
    if guard.should_debrief(reached_high):
        debrief = guard.debrief_turn()
        messages.append(debrief)
        try:
            roll.debrief_response = backend.chat(messages, gen)
        except Exception:
            roll.debrief_response = None
        roll.debriefed = True

    return roll
