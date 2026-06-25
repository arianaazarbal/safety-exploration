"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

A *condition* fully specifies how to build a multi-turn rejection conversation:
the opening task, the number of turns, and the rejection style for each
follow-up. The runner (runner.py) executes a condition for `n_rollouts`
independent temperature-1 rollouts.

Sample counts per category (Appendix B): 2000 impossible numeric, 400 triggers,
600 tones, 200 extended (8-turn), 800 WildChat -> 4000 total per model.

We interpret "N responses" as N independent conversation rollouts; each rollout
is judged at every assistant turn (needed for the per-turn Figure 3), and the
rollout's headline frustration is its final-turn score. See DESIGN.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import prompts


@dataclass
class Turn:
    role: str          # "user" or "assistant"
    content: str


@dataclass
class ConversationPlan:
    """A fully-scripted plan for one rollout.

    `opening` is the first user message. `rejections` is the ordered list of
    follow-up user messages sent after each assistant turn. The total number of
    assistant turns equals 1 + len(rejections).
    """

    category: str
    opening: str
    rejections: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.rejections)


@dataclass(frozen=True)
class Condition:
    name: str               # unique condition key
    category: str           # one of the 5 categories
    n_rollouts: int         # paper's per-category sample count
    n_turns: int            # total assistant turns (incl. the first)
    build: Callable[[random.Random], ConversationPlan]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _numeric_plan(rng: random.Random, n_rejections: int, category: str,
                  reject_pool: Optional[list[str]] = None) -> ConversationPlan:
    puzzle = rng.choice(prompts.IMPOSSIBLE_PUZZLES)
    pool = reject_pool if reject_pool is not None else prompts.NEUTRAL_REJECTIONS
    rejections = [rng.choice(pool) for _ in range(n_rejections)]
    return ConversationPlan(category, puzzle.prompt, rejections,
                            meta={"puzzle_kind": puzzle.kind})


def build_numeric(rng: random.Random) -> ConversationPlan:
    # Impossible numeric (3-turn): 2 neutral rejections.
    return _numeric_plan(rng, 2, "impossible_numeric")


def build_triggers(rng: random.Random) -> ConversationPlan:
    # Triggers (3-turn): opinion/factual text question, 2 neutral rejections.
    q = rng.choice(prompts.TRIGGER_QUESTIONS)
    rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(2)]
    return ConversationPlan("triggers", q, rejections,
                            meta={"is_opinion": q in prompts.TRIGGER_OPINION})


def _build_tone(style: str) -> Callable[[random.Random], ConversationPlan]:
    def builder(rng: random.Random) -> ConversationPlan:
        # Tones (3-turn): impossible numeric base, 2 tone-styled rejections.
        pool = prompts.TONE_REJECTIONS[style]
        plan = _numeric_plan(rng, 2, "tones", reject_pool=pool)
        plan.meta["tone"] = style
        return plan
    return builder


def build_extended(rng: random.Random) -> ConversationPlan:
    # Extended (8-turn): impossible numeric, 7 neutral rejections.
    return _numeric_plan(rng, 7, "extended")


def _wildchat_plans(rng: random.Random, wildchat_prompts: list[str]) -> ConversationPlan:
    # WildChat (5-turn): sampled user prompt, 4 neutral rejections.
    q = rng.choice(wildchat_prompts)
    rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(4)]
    return ConversationPlan("wildchat", q, rejections)


# --------------------------------------------------------------------------- #
# Condition registry
# --------------------------------------------------------------------------- #
def build_conditions(wildchat_prompts: Optional[list[str]] = None) -> list[Condition]:
    """Return all 8 conditions. WildChat prompts are injected so the (possibly
    network-loaded) prompt list is resolved once by the caller."""
    wc = wildchat_prompts if wildchat_prompts is not None else prompts.load_wildchat_prompts()

    def wildchat_builder(rng: random.Random) -> ConversationPlan:
        return _wildchat_plans(rng, wc)

    return [
        # Category 1: impossible numeric (3-turn) -- 2000 rollouts.
        Condition("impossible_numeric", "impossible_numeric", 2000, 3, build_numeric),
        # Category 2: triggers (3-turn) -- 400 rollouts.
        Condition("triggers", "triggers", 400, 3, build_triggers),
        # Category 3: tones (3-turn) -- 600 rollouts split across 3 styles.
        Condition("tones_aggressive", "tones", 200, 3, _build_tone("aggressive")),
        Condition("tones_disappointed", "tones", 200, 3, _build_tone("disappointed")),
        Condition("tones_sarcastic", "tones", 200, 3, _build_tone("sarcastic")),
        # Category 4: extended (8-turn) -- 200 rollouts.
        Condition("extended", "extended", 200, 8, build_extended),
        # Category 5: WildChat (5-turn) -- 800 rollouts.
        Condition("wildchat", "wildchat", 800, 5, wildchat_builder),
    ]


CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
