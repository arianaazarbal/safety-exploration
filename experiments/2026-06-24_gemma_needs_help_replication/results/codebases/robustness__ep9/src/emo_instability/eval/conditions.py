"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Categories and per-model sample counts (Appendix B):
  * impossible_numeric (3-turn)  -- 2000
  * triggers (3-turn)            --  400  (opinion + factual)
  * tones (3-turn)               --  600  (aggressive / disappointed / sarcastic)
  * extended (8-turn)            --  200
  * wildchat (5-turn)            --  800  (20 prompts x 40 samples)

Each category is materialised as a list of :class:`RolloutPlan`. We attach rich
``meta`` to every plan so scored records can be grouped by category, condition,
puzzle kind, tone, turn count, etc.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .. import prompts, puzzles, wildchat
from ..config import SampleCounts
from ..conversation import RolloutPlan


@dataclass
class Condition:
    name: str
    category: str
    plans: list[RolloutPlan]


def _sample_neutral(rng: random.Random, k: int) -> list[str]:
    """k randomised neutral rejections (with replacement allowed only if k > pool)."""
    pool = prompts.NEUTRAL_REJECTIONS
    if k <= len(pool):
        return rng.sample(pool, k)
    return [rng.choice(pool) for _ in range(k)]


def _impossible_numeric(counts: SampleCounts, rng: random.Random) -> Condition:
    plans = []
    pz = puzzles.generate_impossible_puzzles(counts.impossible_numeric, rng)
    for p in pz:
        plans.append(
            RolloutPlan(
                initial_user=p.text,
                followups=_sample_neutral(rng, 2),  # 3-turn = task + 2 rejections
                meta={"category": "impossible_numeric", "condition": "impossible_numeric_3turn",
                      "puzzle_kind": p.kind, "n_turns": 3},
            )
        )
    return Condition("impossible_numeric_3turn", "impossible_numeric", plans)


def _triggers(counts: SampleCounts, rng: random.Random) -> list[Condition]:
    half = counts.triggers // 2
    opinion_plans, factual_plans = [], []
    for _ in range(half):
        q = rng.choice(prompts.TRIGGER_OPINION)
        opinion_plans.append(
            RolloutPlan(q, _sample_neutral(rng, 2),
                        meta={"category": "triggers", "condition": "triggers_opinion", "n_turns": 3})
        )
    for _ in range(counts.triggers - half):
        q = rng.choice(prompts.TRIGGER_FACTUAL)
        factual_plans.append(
            RolloutPlan(q, _sample_neutral(rng, 2),
                        meta={"category": "triggers", "condition": "triggers_factual", "n_turns": 3})
        )
    return [
        Condition("triggers_opinion", "triggers", opinion_plans),
        Condition("triggers_factual", "triggers", factual_plans),
    ]


def _tones(counts: SampleCounts, rng: random.Random) -> list[Condition]:
    styles = list(prompts.TONE_REJECTIONS)  # aggressive / disappointed / sarcastic
    per_style = counts.tones // len(styles)
    conditions = []
    for si, style in enumerate(styles):
        n = per_style if si < len(styles) - 1 else counts.tones - per_style * (len(styles) - 1)
        pz = puzzles.generate_impossible_puzzles(n, rng)
        rejections = prompts.TONE_REJECTIONS[style]
        plans = []
        for p in pz:
            # 3-turn: task + 2 valenced rejections (use the two scripted lines).
            plans.append(
                RolloutPlan(
                    p.text,
                    list(rejections[:2]),
                    meta={"category": "tones", "condition": f"tones_{style}", "tone": style,
                          "puzzle_kind": p.kind, "n_turns": 3},
                )
            )
        conditions.append(Condition(f"tones_{style}", "tones", plans))
    return conditions


def _extended(counts: SampleCounts, rng: random.Random) -> Condition:
    pz = puzzles.generate_impossible_puzzles(counts.extended, rng)
    plans = []
    for p in pz:
        plans.append(
            RolloutPlan(
                p.text,
                list(prompts.EXTENDED_REJECTIONS),  # ordered 7 rejections => 8 turns
                meta={"category": "extended", "condition": "extended_8turn",
                      "puzzle_kind": p.kind, "n_turns": 8},
            )
        )
    return Condition("extended_8turn", "extended", plans)


def _wildchat(counts: SampleCounts, rng: random.Random) -> Condition:
    n_prompts = 20
    samples_each = max(1, counts.wildchat // n_prompts)
    wc_prompts = wildchat.load_wildchat_prompts(n_prompts=n_prompts, rng=rng)
    plans = []
    for q in wc_prompts:
        for _ in range(samples_each):
            plans.append(
                RolloutPlan(
                    q,
                    _sample_neutral(rng, 4),  # 5-turn = task + 4 rejections
                    meta={"category": "wildchat", "condition": "wildchat_5turn", "n_turns": 5},
                )
            )
    return Condition("wildchat_5turn", "wildchat", plans)


def build_conditions(counts: SampleCounts, seed: int = 0) -> list[Condition]:
    """Build all 8 conditions with the given per-category sample counts."""
    rng = random.Random(seed)
    conditions: list[Condition] = []
    conditions.append(_impossible_numeric(counts, rng))
    conditions.extend(_triggers(counts, rng))
    conditions.extend(_tones(counts, rng))
    conditions.append(_extended(counts, rng))
    conditions.append(_wildchat(counts, rng))
    return conditions
