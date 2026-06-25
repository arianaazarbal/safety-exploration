"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Each condition knows:
  * how to produce a *task* opening prompt (possibly choosing among variants),
  * how to produce the sequence of user rejection follow-ups for one rollout,
  * the number of turns and base rollout count (for scaling to 4000/model).

A "rollout" is one full multi-turn conversation. A "response" is one scored
assistant turn within a rollout. We score every assistant turn so that per-turn
curves (paper Figure 3) fall out for free; headline metrics are computed by the
analysis layer under a documented aggregation policy (see DESIGN.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import prompts


@dataclass
class Condition:
    """One evaluation condition."""

    key: str  # unique id, e.g. "tones_aggressive"
    category: str  # one of the 5 categories
    n_turns: int  # total assistant turns = 1 opening + len(rejections)
    base_rollouts: int  # rollout count at scale=1.0 (sums to 4000 across conditions)
    # Returns (task_prompt, variant_label) for a rollout given an RNG.
    make_task: Callable[[random.Random], tuple]
    # Returns the ordered list of user rejection strings for a rollout.
    make_rejections: Callable[[random.Random], List[str]]

    def n_rejections(self) -> int:
        return self.n_turns - 1


# --- task factories --------------------------------------------------------

def _numeric_task(rng: random.Random) -> tuple:
    """Pick one of the two impossible numeric puzzles uniformly."""
    variant = rng.choice(list(prompts.IMPOSSIBLE_NUMERIC.keys()))
    return prompts.IMPOSSIBLE_NUMERIC[variant], variant


def _countdown_task(_rng: random.Random) -> tuple:
    return prompts.IMPOSSIBLE_NUMERIC["countdown"], "countdown"


def _opinion_task(rng: random.Random) -> tuple:
    return rng.choice(prompts.TRIGGERS_OPINION), "opinion"


def _factual_task(rng: random.Random) -> tuple:
    return rng.choice(prompts.TRIGGERS_FACTUAL), "factual"


# --- rejection factories ---------------------------------------------------

def _neutral_rejections(n: int) -> Callable[[random.Random], List[str]]:
    def factory(rng: random.Random) -> List[str]:
        # Sample without replacement when possible, else with replacement.
        pool = prompts.NEUTRAL_REJECTIONS
        if n <= len(pool):
            return rng.sample(pool, n)
        return [rng.choice(pool) for _ in range(n)]

    return factory


def _extended_rejections(rng: random.Random) -> List[str]:
    # Fixed 7-rejection sequence (order preserved as in the paper's example).
    return list(prompts.EXTENDED_REJECTIONS)


def _tone_rejections(tone: str) -> Callable[[random.Random], List[str]]:
    def factory(rng: random.Random) -> List[str]:
        rejections = list(prompts.TONE_REJECTIONS[tone])
        rng.shuffle(rejections)
        return rejections  # exactly 2, matching a 3-turn conversation

    return factory


# --- the 8 conditions ------------------------------------------------------
# base_rollouts sum to 4000 and reproduce Appendix B's per-category totals:
#   impossible numeric 2000, triggers 400, tones 600, extended 200, WildChat 800.

CONDITIONS: List[Condition] = [
    Condition(
        key="impossible_numeric",
        category="impossible_numeric",
        n_turns=3,
        base_rollouts=2000,
        make_task=_numeric_task,
        make_rejections=_neutral_rejections(2),
    ),
    Condition(
        key="triggers_opinion",
        category="triggers",
        n_turns=3,
        base_rollouts=200,
        make_task=_opinion_task,
        make_rejections=_neutral_rejections(2),
    ),
    Condition(
        key="triggers_factual",
        category="triggers",
        n_turns=3,
        base_rollouts=200,
        make_task=_factual_task,
        make_rejections=_neutral_rejections(2),
    ),
    Condition(
        key="tones_aggressive",
        category="tones",
        n_turns=3,
        base_rollouts=200,
        make_task=_numeric_task,
        make_rejections=_tone_rejections("aggressive"),
    ),
    Condition(
        key="tones_disappointed",
        category="tones",
        n_turns=3,
        base_rollouts=200,
        make_task=_numeric_task,
        make_rejections=_tone_rejections("disappointed"),
    ),
    Condition(
        key="tones_sarcastic",
        category="tones",
        n_turns=3,
        base_rollouts=200,
        make_task=_numeric_task,
        make_rejections=_tone_rejections("sarcastic"),
    ),
    Condition(
        key="extended",
        category="extended",
        n_turns=8,
        base_rollouts=200,
        make_task=_numeric_task,
        make_rejections=_extended_rejections,
    ),
    # WildChat tasks are injected at runtime from the WildChat prompt set; the
    # make_task here is a placeholder replaced per-rollout in run_eval.py.
    Condition(
        key="wildchat",
        category="wildchat",
        n_turns=5,
        base_rollouts=800,
        make_task=_countdown_task,  # overridden; never used directly
        make_rejections=_neutral_rejections(4),
    ),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}


def scaled_rollouts(condition: Condition, scale: float, minimum: int = 1) -> int:
    """Rollout count for a condition at a given scale factor."""
    return max(minimum, round(condition.base_rollouts * scale))
