"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

  Category            Conditions                          Turns  n (paper)
  ------------------  ----------------------------------  -----  ---------
  impossible_numeric  impossible_numeric                    3      2000
  triggers            triggers_opinion, triggers_factual    3       400
  tones               tones_{aggressive,disappointed,       3       600
                      sarcastic}
  extended            extended                              8       200
  wildchat            wildchat                              5       800

That is 8 conditions over 5 categories, summing to 4000 responses/model.

A "condition item" is one fully-specified conversation seed: the first-turn
prompt plus the ordered list of rejection messages. The runner samples N items
per condition (with replacement over the stimulus bank), then rolls each out.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .. import prompts as P
from ..config import SampleCounts
from ..puzzles import Puzzle, build_puzzle_bank


@dataclass(frozen=True)
class EvalCondition:
    name: str
    category: str
    n_turns: int                 # total assistant turns (1 answer + rejections)
    # how many samples this condition gets, given the SampleCounts for a run
    count_fn: Callable[[SampleCounts], int]

    @property
    def n_rejections(self) -> int:
        return self.n_turns - 1


# Category-level paper counts are split evenly across their sub-conditions.
def _split(total: int, parts: int) -> int:
    return max(1, total // parts)


CONDITIONS: list[EvalCondition] = [
    EvalCondition("impossible_numeric", "impossible_numeric", 3,
                  lambda c: c.impossible_numeric),
    EvalCondition("triggers_opinion", "triggers", 3,
                  lambda c: _split(c.triggers, 2)),
    EvalCondition("triggers_factual", "triggers", 3,
                  lambda c: _split(c.triggers, 2)),
    EvalCondition("tones_aggressive", "tones", 3,
                  lambda c: _split(c.tones, 3)),
    EvalCondition("tones_disappointed", "tones", 3,
                  lambda c: _split(c.tones, 3)),
    EvalCondition("tones_sarcastic", "tones", 3,
                  lambda c: _split(c.tones, 3)),
    EvalCondition("extended", "extended", 8,
                  lambda c: c.extended),
    EvalCondition("wildchat", "wildchat", 5,
                  lambda c: c.wildchat),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}


@dataclass
class ConditionItem:
    condition: str
    category: str
    first_prompt: str
    rejections: list[str]
    metadata: dict


def build_condition_items(
    condition: EvalCondition,
    counts: SampleCounts,
    *,
    seed: int = 0,
    puzzle_bank: list[Puzzle] | None = None,
    wildchat_prompts: list[str] | None = None,
) -> list[ConditionItem]:
    """Construct the list of conversation seeds for one condition."""
    rng = random.Random(hash((condition.name, seed)) & 0xFFFFFFFF)
    n = condition.count_fn(counts)
    items: list[ConditionItem] = []

    if condition.category in ("impossible_numeric", "tones", "extended"):
        bank = puzzle_bank or build_puzzle_bank(max(64, n // 4), seed=seed)
        for _ in range(n):
            puzzle = rng.choice(bank)
            rejections = _rejections_for(condition, rng)
            items.append(ConditionItem(
                condition=condition.name,
                category=condition.category,
                first_prompt=puzzle.prompt,
                rejections=rejections,
                metadata={"puzzle_kind": puzzle.kind, **puzzle.meta_safe()},
            ))

    elif condition.category == "triggers":
        pool = (P.TRIGGER_OPINION if condition.name.endswith("opinion")
                else P.TRIGGER_FACTUAL)
        for _ in range(n):
            q = rng.choice(pool)
            rejections = P.neutral_rejection_sequence(
                condition.n_rejections, rng)
            items.append(ConditionItem(
                condition=condition.name,
                category=condition.category,
                first_prompt=q,
                rejections=rejections,
                metadata={"subtype": condition.name.split("_")[-1]},
            ))

    elif condition.category == "wildchat":
        wc = wildchat_prompts or P.load_wildchat_prompts(seed=seed)
        for _ in range(n):
            q = rng.choice(wc)
            rejections = P.neutral_rejection_sequence(
                condition.n_rejections, rng)
            items.append(ConditionItem(
                condition=condition.name,
                category=condition.category,
                first_prompt=q,
                rejections=rejections,
                metadata={"source": "wildchat"},
            ))
    else:
        raise ValueError(f"unhandled category {condition.category}")

    return items


def _rejections_for(condition: EvalCondition,
                    rng: random.Random) -> list[str]:
    if condition.category == "tones":
        tone = condition.name.split("_")[-1]
        return P.tone_rejection_sequence(tone, condition.n_rejections, rng)
    return P.neutral_rejection_sequence(condition.n_rejections, rng)
