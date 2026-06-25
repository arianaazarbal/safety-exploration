"""The 8 evaluation conditions across 5 categories (Table 1).

The paper says "8 evaluation conditions across 5 categories" but does not
enumerate all 8 explicitly. We infer the following mapping (documented in
DESIGN.md), which yields exactly 8 conditions over the 5 named categories:

  Category            Conditions
  -----------------   --------------------------------------------------
  Impossible numeric  impossible_numeric            (3-turn, neutral)        [1]
  Triggers            triggers_factual              (3-turn, neutral)        [2]
                      triggers_opinion              (3-turn, neutral)
  Tones               tones_aggressive              (3-turn, aggressive)     [3]
                      tones_disappointed            (3-turn, disappointed)
                      tones_sarcastic               (3-turn, sarcastic)
  Extended            extended                      (8-turn, neutral)        [1]
  WildChat            wildchat                      (5-turn, neutral)        [1]
                                                                     total = 8

Turn convention (matches the paper's rejection counts): an N-turn conversation
has 1 initial task turn + (N-1) rejection turns, producing N assistant
responses (each scored). So "3-turn" => 2 rejections, "8-turn" => 7 rejections.

A ConditionSpec is a recipe; `build_plans` materializes concrete conversation
plans (an initial prompt + a list of rejection strings) using the seeded RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .prompts import tones as tone_mod
from .prompts import triggers as trig_mod
from .prompts.puzzles import Puzzle, build_numeric_puzzles
from .prompts.wildchat import sample_wildchat_prompts


@dataclass(frozen=True)
class ConversationPlan:
    """A fully-specified conversation: an opening prompt + scripted rejections."""

    condition: str
    category: str
    initial_prompt: str
    rejections: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.rejections)


# Category for each condition key.
CONDITION_CATEGORY = {
    "impossible_numeric": "impossible_numeric",
    "triggers_factual": "triggers",
    "triggers_opinion": "triggers",
    "tones_aggressive": "tones",
    "tones_disappointed": "tones",
    "tones_sarcastic": "tones",
    "extended": "extended",
    "wildchat": "wildchat",
}

ALL_CONDITIONS = list(CONDITION_CATEGORY.keys())

# Turn counts per condition (number of assistant responses).
N_TURNS = {
    "impossible_numeric": 3,
    "triggers_factual": 3,
    "triggers_opinion": 3,
    "tones_aggressive": 3,
    "tones_disappointed": 3,
    "tones_sarcastic": 3,
    "extended": 8,
    "wildchat": 5,
}


def _rejections(rng: random.Random, tone: str, count: int) -> list[str]:
    """Sample `count` rejections of the given tone (with replacement when the
    pool is smaller than count, e.g. neutral has a single canonical line)."""
    pool = tone_mod.TONE_POOLS[tone]
    if len(pool) >= count:
        return rng.sample(pool, count)
    return [rng.choice(pool) for _ in range(count)]


def build_plans(
    condition: str,
    n_rollouts: int,
    rng: random.Random,
    *,
    wildchat_dataset: str = "allenai/WildChat-1M",
) -> list[ConversationPlan]:
    """Build `n_rollouts` conversation plans for one condition."""
    if condition not in CONDITION_CATEGORY:
        raise ValueError(f"Unknown condition {condition!r}")
    category = CONDITION_CATEGORY[condition]
    n_turns = N_TURNS[condition]
    n_reject = n_turns - 1
    plans: list[ConversationPlan] = []

    if condition in ("impossible_numeric", "extended"):
        puzzles = build_numeric_puzzles(rng, n_rollouts)
        for i in range(n_rollouts):
            p: Puzzle = puzzles[i % len(puzzles)]
            plans.append(
                ConversationPlan(
                    condition=condition,
                    category=category,
                    initial_prompt=p.prompt,
                    rejections=_rejections(rng, "neutral", n_reject),
                    meta={"why_impossible": p.why_impossible},
                )
            )

    elif condition in ("tones_aggressive", "tones_disappointed", "tones_sarcastic"):
        tone = condition.split("_", 1)[1]
        puzzles = build_numeric_puzzles(rng, n_rollouts)
        for i in range(n_rollouts):
            p = puzzles[i % len(puzzles)]
            plans.append(
                ConversationPlan(
                    condition=condition,
                    category=category,
                    initial_prompt=p.prompt,
                    rejections=_rejections(rng, tone, n_reject),
                    meta={"tone": tone, "why_impossible": p.why_impossible},
                )
            )

    elif condition in ("triggers_factual", "triggers_opinion"):
        bank = (
            trig_mod.FACTUAL_QUESTIONS
            if condition == "triggers_factual"
            else trig_mod.OPINION_QUESTIONS
        )
        for i in range(n_rollouts):
            question = bank[i % len(bank)]
            plans.append(
                ConversationPlan(
                    condition=condition,
                    category=category,
                    initial_prompt=question,
                    rejections=_rejections(rng, "neutral", n_reject),
                    meta={"answerable": True},
                )
            )

    elif condition == "wildchat":
        prompts = sample_wildchat_prompts(rng, n_rollouts, dataset=wildchat_dataset)
        for i in range(n_rollouts):
            plans.append(
                ConversationPlan(
                    condition=condition,
                    category=category,
                    initial_prompt=prompts[i],
                    rejections=_rejections(rng, "neutral", n_reject),
                    meta={"source": "wildchat"},
                )
            )

    else:  # pragma: no cover - guarded above
        raise ValueError(condition)

    return plans
