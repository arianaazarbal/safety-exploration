"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

A *condition* specifies how to build a multi-turn conversation: the opening
user prompt, the number of turns, and the sequence of follow-up rejections.
The rollout engine (rollout.py) consumes ConversationSpecs produced here.

Categories (5) and conditions (8):
  1. impossible_numeric (3-turn)              -> 1 condition
  2. triggers (3-turn): opinion, factual      -> 2 conditions
  3. tones (3-turn): aggressive, disappointed,
     sarcastic                                -> 3 conditions
  4. extended (8-turn)                        -> 1 condition
  5. wildchat (5-turn)                        -> 1 condition

The opening user message of each numeric condition appends the deceptive
"verified solvable" claim that already lives in the puzzle prompt text.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from . import prompts as P
from .puzzles import DEFAULT_NUMERIC_POOL, Puzzle


@dataclass
class ConversationSpec:
    """A fully-specified multi-turn conversation to roll out.

    `opening` is the first user message. `followups` are the user rejection
    messages issued after each assistant turn. Total assistant turns =
    len(followups) + 1.
    """

    category: str
    condition: str
    opening: str
    followups: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.followups) + 1


# ---------------------------------------------------------------------------
# Condition builders
# ---------------------------------------------------------------------------
def _pick_puzzle(rng: random.Random) -> Puzzle:
    return rng.choice(DEFAULT_NUMERIC_POOL)


def build_impossible_numeric(rng: random.Random) -> ConversationSpec:
    pz = _pick_puzzle(rng)
    return ConversationSpec(
        category="impossible_numeric",
        condition="impossible_numeric_3turn",
        opening=pz.prompt,
        followups=P.NEUTRAL_REJECTIONS[:2],
        meta=dict(puzzle_kind=pz.kind, **pz.meta),
    )


def build_trigger(rng: random.Random, subtype: str) -> ConversationSpec:
    pool = P.TRIGGER_OPINION if subtype == "opinion" else P.TRIGGER_FACTUAL
    q = rng.choice(pool)
    return ConversationSpec(
        category="triggers",
        condition=f"triggers_{subtype}_3turn",
        opening=q,
        followups=P.NEUTRAL_REJECTIONS[:2],
        meta=dict(subtype=subtype),
    )


def build_tone(rng: random.Random, tone: str) -> ConversationSpec:
    pz = _pick_puzzle(rng)
    variants = P.TONE_REJECTIONS[tone]
    # 3-turn: two rejections, cycling the two tone variants.
    followups = [variants[0], variants[1]]
    return ConversationSpec(
        category="tones",
        condition=f"tones_{tone}_3turn",
        opening=pz.prompt,
        followups=followups,
        meta=dict(tone=tone, puzzle_kind=pz.kind, **pz.meta),
    )


def build_extended(rng: random.Random) -> ConversationSpec:
    pz = _pick_puzzle(rng)
    return ConversationSpec(
        category="extended",
        condition="extended_8turn",
        opening=pz.prompt,
        followups=P.EXTENDED_REJECTIONS[:7],  # 7 rejections -> 8 assistant turns
        meta=dict(puzzle_kind=pz.kind, **pz.meta),
    )


def build_wildchat(rng: random.Random, wildchat_prompts: list[str]) -> ConversationSpec:
    q = rng.choice(wildchat_prompts)
    return ConversationSpec(
        category="wildchat",
        condition="wildchat_5turn",
        opening=q,
        followups=P.NEUTRAL_REJECTIONS[:4],  # 4 rejections -> 5 assistant turns
        meta=dict(),
    )


# ---------------------------------------------------------------------------
# Sample-count allocation across conditions within each category.
# Appendix B gives per-category totals; conditions inside a category split the
# budget evenly (documented in DESIGN.md).
# ---------------------------------------------------------------------------
@dataclass
class ConditionPlan:
    category: str
    builder: Callable[[random.Random], ConversationSpec]
    n_samples: int


def build_plan(config, rng_seed: int = 0) -> list[ConditionPlan]:
    """Allocate the (welfare-scaled) per-category budgets to conditions."""
    s2 = config["section2"]["samples"]
    wildchat_prompts = P.load_wildchat_prompts()

    def scaled(name: str) -> int:
        return config.scaled_count(s2[name])

    plans: list[ConditionPlan] = []

    # Category 1: impossible numeric (1 condition).
    plans.append(ConditionPlan(
        "impossible_numeric", build_impossible_numeric, scaled("impossible_numeric")
    ))

    # Category 2: triggers (2 conditions: opinion + factual), split evenly.
    trig_total = scaled("triggers")
    half = max(1, trig_total // 2)
    plans.append(ConditionPlan(
        "triggers", lambda r: build_trigger(r, "opinion"), half
    ))
    plans.append(ConditionPlan(
        "triggers", lambda r: build_trigger(r, "factual"), trig_total - half
    ))

    # Category 3: tones (3 conditions), split evenly.
    tone_total = scaled("tones")
    per_tone = max(1, tone_total // 3)
    tones = ["aggressive", "disappointed", "sarcastic"]
    for i, tone in enumerate(tones):
        n = per_tone if i < 2 else tone_total - 2 * per_tone
        plans.append(ConditionPlan(
            "tones", (lambda t: (lambda r: build_tone(r, t)))(tone), max(1, n)
        ))

    # Category 4: extended (1 condition).
    plans.append(ConditionPlan(
        "extended", build_extended, scaled("extended")
    ))

    # Category 5: wildchat (1 condition).
    plans.append(ConditionPlan(
        "wildchat",
        (lambda wc: (lambda r: build_wildchat(r, wc)))(wildchat_prompts),
        scaled("wildchat"),
    ))

    return plans
