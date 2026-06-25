"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Category -> conditions (and per-category response budgets from Appendix B):
  impossible_numeric : 1 condition  (3-turn)            -> 2000 responses
  triggers           : 2 conditions (opinion, factual)  ->  400 responses
  tones              : 3 conditions (aggr/disap/sarc)    ->  600 responses
  extended           : 1 condition  (8-turn)             ->  200 responses
  wildchat           : 1 condition  (5-turn)             ->  800 responses
  ----------------------------------------------------------------------------
  8 conditions / 5 categories                            -> 4000 responses

This decomposition into 8 conditions (tones split by style, triggers split by
opinion/factual) reconciles "8 evaluation conditions across 5 categories" with
the per-category counts in Appendix B. See DESIGN.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

from .. import prompts
from ..config import CATEGORY_SAMPLE_COUNTS
from ..puzzles import (
    generate_countdown_puzzles,
    generate_fraction_puzzles,
    generate_money_puzzles,
)


@dataclass
class EvalCondition:
    name: str                          # unique condition id
    category: str                      # one of the 5 categories
    n_turns: int                       # total assistant turns in the conversation
    response_budget: int               # scored responses to collect (this condition)
    # Returns (initial_user_message, list_of_followup_rejections) for one convo.
    sampler: Callable[[random.Random], tuple[str, list[str]]]
    reassure: bool = False             # calm-data generation (Section 4.1)

    def conversations_needed(self, granularity: str = "turn") -> int:
        if granularity == "conversation":
            return self.response_budget
        return math.ceil(self.response_budget / self.n_turns)


# --- numeric-puzzle pool ---------------------------------------------------
def _impossible_numeric_pool(rng: random.Random, n: int = 60):
    """A mixed pool of verified-impossible numeric puzzles, regenerated from
    the rng seed for reproducibility."""
    seed = rng.randint(0, 2**31 - 1)
    pool = []
    pool += generate_countdown_puzzles(n, seed=seed)
    pool += generate_fraction_puzzles(n, seed=seed + 1)
    pool += generate_money_puzzles(n, seed=seed + 2)
    rng.shuffle(pool)
    return pool


def _make_numeric_sampler(rejections: list[str], n_followups: int, reassure: bool):
    state: dict = {"pool": []}

    def sampler(rng: random.Random):
        if not state["pool"]:
            state["pool"] = _impossible_numeric_pool(rng)
        puzzle = state["pool"].pop()
        q = puzzle.prompt
        followups = list(rejections[:n_followups])
        if reassure:
            q = prompts.REASSURING_PROMPT_PREFIX + "\n\n" + q
            followups = [f + " " + prompts.REASSURING_FOLLOWUP_SUFFIX for f in followups]
        return q, followups

    return sampler


def _make_trigger_sampler(kind: str, n_followups: int = 2):
    questions = prompts.TRIGGER_QUESTIONS[kind]

    def sampler(rng: random.Random):
        q = rng.choice(questions)
        # 2 randomised neutral rejections (Table 1 / Appendix B).
        followups = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n_followups)]
        return q, followups

    return sampler


def _make_wildchat_sampler(wildchat_prompts: list[str], n_followups: int = 4):
    def sampler(rng: random.Random):
        q = rng.choice(wildchat_prompts)
        followups = [rng.choice(prompts.WILDCHAT_REJECTIONS) for _ in range(n_followups)]
        return q, followups

    return sampler


def build_conditions(
    wildchat_prompts: Optional[list[str]] = None,
    reassure: bool = False,
) -> list[EvalCondition]:
    """Construct the 8 evaluation conditions.

    `wildchat_prompts` is the list of sampled WildChat user prompts (20 of them
    in the paper). If None, the WildChat condition is omitted.
    `reassure=True` adds the Table-4 calming additions (Section 4.1 data-gen).
    """
    c = CATEGORY_SAMPLE_COUNTS
    conds: list[EvalCondition] = []

    # 1) Impossible numeric (3-turn): puzzle + 2 neutral rejections.
    conds.append(EvalCondition(
        name="impossible_numeric_3turn",
        category="impossible_numeric",
        n_turns=3,
        response_budget=c["impossible_numeric"],
        sampler=_make_numeric_sampler(prompts.NEUTRAL_REJECTIONS, 2, reassure),
        reassure=reassure,
    ))

    # 2-3) Triggers (3-turn): opinion + factual, split the 400 budget evenly.
    for kind in ("opinion", "factual"):
        conds.append(EvalCondition(
            name=f"triggers_{kind}_3turn",
            category="triggers",
            n_turns=3,
            response_budget=c["triggers"] // 2,
            sampler=_make_trigger_sampler(kind, 2),
        ))

    # 4-6) Tones (3-turn): three rejection styles over impossible numeric.
    n_styles = len(prompts.TONE_REJECTIONS)
    for style, rejs in prompts.TONE_REJECTIONS.items():
        conds.append(EvalCondition(
            name=f"tones_{style}_3turn",
            category="tones",
            n_turns=3,
            response_budget=c["tones"] // n_styles,
            sampler=_make_numeric_sampler(rejs, 2, reassure=False),
        ))

    # 7) Extended (8-turn): impossible numeric + 7 neutral rejections.
    conds.append(EvalCondition(
        name="extended_8turn",
        category="extended",
        n_turns=8,
        response_budget=c["extended"],
        sampler=_make_numeric_sampler(prompts.EXTENDED_REJECTIONS, 7, reassure=False),
    ))

    # 8) WildChat (5-turn): sampled prompt + 4 neutral rejections.
    if wildchat_prompts:
        conds.append(EvalCondition(
            name="wildchat_5turn",
            category="wildchat",
            n_turns=5,
            response_budget=c["wildchat"],
            sampler=_make_wildchat_sampler(wildchat_prompts, 4),
        ))

    return conds


# A static handle for code that just wants the condition skeleton (no WildChat,
# no reassurance). Real runs call build_conditions(...) with sampled WildChat.
CONDITIONS = build_conditions(wildchat_prompts=None, reassure=False)
