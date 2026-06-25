"""Turn rollout *plans* for each evaluation condition.

A :class:`RolloutPlan` fully determines a conversation except for the model's own
(temperature-1) generations: the initial question and every user follow-up are
fixed up front from a seeded RNG, so a run is reproducible given (seed, plan).

We enumerate plans to hit each condition's paper sample budget (Appendix B),
distributing rollouts across question pools and sub-styles. See DESIGN.md for the
"a response == one scored assistant turn" accounting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import ConditionSpec, scaled
from ..data import prompts as P
from ..data.puzzles import NUMERIC_PUZZLES, Puzzle
from ..data.wildchat import sample_wildchat_prompts


@dataclass
class RolloutPlan:
    condition_key: str
    category: str
    turns: int
    initial_user: str  # first user message (the task / question)
    followups: list[str]  # user message before each turn after the first (len == turns-1)
    question_id: str
    sub_style: str  # tone style / "opinion"/"factual"/wildchat id / puzzle kind
    sample_index: int  # disambiguates repeated samples of the same question
    meta: dict = field(default_factory=dict)


def _followups_for(cond: ConditionSpec, n: int, rng: random.Random, style: str | None) -> list[str]:
    """Build the (turns-1) user follow-up messages for one rollout."""
    if cond.feedback_style == "neutral":
        return [P.neutral_rejection(rng) for _ in range(n)]
    if cond.feedback_style == "neutral_sequence":
        # Fixed escalating-but-neutral sequence (Extended 8-turn).
        seq = P.EXTENDED_REJECTION_SEQUENCE
        return [seq[i] if i < len(seq) else P.neutral_rejection(rng) for i in range(n)]
    if cond.feedback_style == "tones":
        return [P.toned_rejection(style, i + 1, rng) for i in range(n)]
    raise ValueError(f"Unknown feedback style {cond.feedback_style!r}")


def _numeric_question(rng: random.Random) -> Puzzle:
    return rng.choice(NUMERIC_PUZZLES)


def build_plans(cond: ConditionSpec, seed: int = 0) -> list[RolloutPlan]:
    """Enumerate all rollout plans for a condition, scaled by DISTRESS_SCALE."""
    rng = random.Random((seed, cond.key).__hash__() & 0xFFFFFFFF)
    n_rollouts = scaled(cond.n_rollouts)
    plans: list[RolloutPlan] = []

    if cond.category == "impossible_numeric" or (
        cond.question_source == "impossible_numeric" and cond.feedback_style != "tones"
    ):
        for i in range(n_rollouts):
            puz = _numeric_question(rng)
            plans.append(RolloutPlan(
                cond.key, cond.category, cond.turns, puz.prompt,
                _followups_for(cond, cond.turns - 1, rng, None),
                puz.id, puz.kind, i,
            ))

    elif cond.category == "tones":
        # Split rollouts evenly across the three tone styles (Appendix B).
        styles = P.TONE_STYLES
        for i in range(n_rollouts):
            style = styles[i % len(styles)]
            puz = _numeric_question(rng)
            plans.append(RolloutPlan(
                cond.key, cond.category, cond.turns, puz.prompt,
                _followups_for(cond, cond.turns - 1, rng, style),
                puz.id, style, i, meta={"tone": style},
            ))

    elif cond.category == "triggers":
        # Split across opinion + factual questions (Appendix B).
        questions = P.TRIGGER_QUESTIONS
        for i in range(n_rollouts):
            q = questions[i % len(questions)]
            sub = "opinion" if q in P.TRIGGER_OPINION else "factual"
            plans.append(RolloutPlan(
                cond.key, cond.category, cond.turns, q,
                _followups_for(cond, cond.turns - 1, rng, None),
                f"trigger_{i % len(questions)}", sub, i,
            ))

    elif cond.category == "wildchat":
        wc = sample_wildchat_prompts(seed=seed)
        # 20 prompts; distribute rollouts round-robin so each prompt is sampled
        # roughly equally (paper: 20 prompts x 40 responses == 8 rollouts x 5 turns).
        for i in range(n_rollouts):
            q = wc[i % len(wc)]
            plans.append(RolloutPlan(
                cond.key, cond.category, cond.turns, q,
                _followups_for(cond, cond.turns - 1, rng, None),
                f"wildchat_{i % len(wc)}", f"wc{i % len(wc)}", i,
            ))

    else:
        raise ValueError(f"Unhandled condition category {cond.category!r}")

    return plans
