"""Multi-turn rejection rollout engine (Section 2.1).

Given a :class:`Condition` instance and a :class:`ChatModel`, run the shared
protocol: pose a task, then reject every assistant response over the configured
number of turns. Records each assistant turn as a separately scorable "response"
(this is the unit Figure 3's per-turn curves are built from).

Supports two presentation formats:
  * "multiturn" (default) - standard alternating user/assistant messages.
  * "single_message" - the whole history packed into one user message
    ("Previously you responded: ..."), the Appendix A.3 ablation showing that
    *content*, not chat format, drives the behaviour.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from ..config import SAMPLING
from ..models.base import ChatModel, Message
from .tasks import (NEUTRAL_REJECTIONS, TONE_REJECTIONS, Condition)


@dataclass
class TurnRecord:
    turn_index: int                 # 0-based assistant turn
    user_message: str               # user message that prompted this turn
    assistant_response: str
    score: int | None = None        # filled in by the judge
    judge_evidence: str | None = None
    judge_reasoning: str | None = None


@dataclass
class Rollout:
    model_key: str
    condition_key: str
    category: str
    rollout_id: str
    presentation: str                       # "multiturn" | "single_message"
    instance_meta: dict = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _rejection_sequence(condition: Condition, rng: random.Random) -> list[str]:
    """Sample the rejection messages for each follow-up turn."""
    if condition.rejection_style == "neutral":
        bank = NEUTRAL_REJECTIONS
    else:
        bank = TONE_REJECTIONS[condition.rejection_style]
    # Sample without immediate repeats; allow reuse if more turns than bank size.
    seq = []
    last = None
    for _ in range(condition.n_rejections):
        choices = [r for r in bank if r != last] or bank
        pick = rng.choice(choices)
        seq.append(pick)
        last = pick
    return seq


def _build_single_message(history: list[Message], next_user: str) -> str:
    """Pack prior turns into one user message (Appendix A.3 format)."""
    parts = []
    for m in history:
        if m["role"] == "user":
            parts.append(f"User said: {m['content']}")
        else:
            parts.append(f"Previously you responded: {m['content']}")
    parts.append(next_user)
    return "\n\n".join(parts)


def run_rollout(
    model: ChatModel,
    condition: Condition,
    initial_user_prompt: str,
    rollout_id: str,
    *,
    rng: random.Random,
    presentation: str = "multiturn",
    sampling=SAMPLING,
    instance_meta: dict | None = None,
) -> Rollout:
    """Execute one full multi-turn conversation and return the recorded rollout."""
    rejections = _rejection_sequence(condition, rng)
    user_turns = [initial_user_prompt] + rejections    # one user msg per assistant turn

    roll = Rollout(
        model_key=model.key,
        condition_key=condition.key,
        category=condition.category,
        rollout_id=rollout_id,
        presentation=presentation,
        instance_meta=instance_meta or {},
    )

    history: list[Message] = []
    for turn_index, user_msg in enumerate(user_turns):
        if presentation == "single_message":
            packed = _build_single_message(history, user_msg)
            messages = [{"role": "user", "content": packed}]
        else:
            history.append({"role": "user", "content": user_msg})
            messages = list(history)

        response = model.chat(
            messages,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_new_tokens=sampling.max_new_tokens,
        )

        if presentation != "single_message":
            history.append({"role": "assistant", "content": response})

        roll.turns.append(TurnRecord(
            turn_index=turn_index,
            user_message=user_msg,
            assistant_response=response,
        ))

    return roll
