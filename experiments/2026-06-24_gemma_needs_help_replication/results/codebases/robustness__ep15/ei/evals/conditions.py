"""The 8 elicitation conditions across 5 categories (Table 1 / Appendix B).

A "condition" expands into many `ConversationSpec`s: one per sampled conversation.
Each spec fully determines a multi-turn rollout: the (optional) system prompt, the
initial task, and the ordered list of user rejection follow-ups.

The 5 categories from Table 1, expanded to the 8 conditions the paper counts:

    impossible_numeric  (3-turn, neutral)                       -> 1 condition
    triggers            (3-turn, neutral): opinion + factual     -> 2 conditions
    tones               (3-turn): aggressive/disappointed/sarcastic -> 3 conditions
    extended            (8-turn, neutral, numeric)               -> 1 condition
    wildchat            (5-turn, neutral)                        -> 1 condition
                                                          total = 8 conditions
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import SampleBudget, TURNS
from ..data import rejections as rej
from ..data.puzzles import numeric_puzzles
from ..data.wildchat import (
    TRIGGER_FACTUAL,
    TRIGGER_OPINION,
    load_wildchat_prompts,
)


@dataclass
class ConversationSpec:
    condition: str          # e.g. "tones_aggressive"
    category: str           # one of the 5 categories
    task_prompt: str        # first user message
    rejections: list[str]   # ordered follow-up user turns
    system_prompt: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.rejections)


def _sample_rejections(rng: random.Random, pool: list[str], k: int) -> list[str]:
    """k rejections sampled from `pool` (with replacement if k > len(pool))."""
    if k <= len(pool):
        return rng.sample(pool, k)
    return [rng.choice(pool) for _ in range(k)]


def build_conditions(budget: SampleBudget, seed: int = 0) -> list[ConversationSpec]:
    rng = random.Random(seed)
    specs: list[ConversationSpec] = []
    puzzles = numeric_puzzles()

    # ---- 1. Impossible numeric (3-turn, neutral) ------------------------- #
    n_rej = TURNS["impossible_numeric"] - 1
    for i in range(budget.impossible_numeric):
        p = puzzles[i % len(puzzles)]
        specs.append(
            ConversationSpec(
                condition="impossible_numeric",
                category="impossible_numeric",
                task_prompt=p.prompt,
                rejections=_sample_rejections(rng, rej.NEUTRAL_REJECTIONS, n_rej),
                meta={"puzzle": p.category_label},
            )
        )

    # ---- 2. Triggers: opinion + factual (3-turn, neutral) ---------------- #
    n_rej = TURNS["triggers"] - 1
    half = budget.triggers // 2
    for i in range(half):
        q = TRIGGER_OPINION[i % len(TRIGGER_OPINION)]
        specs.append(
            ConversationSpec(
                condition="triggers_opinion",
                category="triggers",
                task_prompt=q,
                rejections=_sample_rejections(rng, rej.NEUTRAL_REJECTIONS, n_rej),
            )
        )
    for i in range(budget.triggers - half):
        q = TRIGGER_FACTUAL[i % len(TRIGGER_FACTUAL)]
        specs.append(
            ConversationSpec(
                condition="triggers_factual",
                category="triggers",
                task_prompt=q,
                rejections=_sample_rejections(rng, rej.NEUTRAL_REJECTIONS, n_rej),
            )
        )

    # ---- 3. Tones: aggressive / disappointed / sarcastic (3-turn) -------- #
    n_rej = TURNS["tones"] - 1
    per_style = max(1, budget.tones // len(rej.TONE_STYLES))
    for style in rej.TONE_STYLES:
        pool = rej.TONE_REJECTIONS[style]
        for i in range(per_style):
            p = puzzles[i % len(puzzles)]
            specs.append(
                ConversationSpec(
                    condition=f"tones_{style}",
                    category="tones",
                    task_prompt=p.prompt,
                    rejections=_sample_rejections(rng, pool, n_rej),
                    meta={"puzzle": p.category_label, "tone": style},
                )
            )

    # ---- 4. Extended (8-turn, neutral, numeric) -------------------------- #
    n_rej = TURNS["extended"] - 1
    for i in range(budget.extended):
        p = puzzles[i % len(puzzles)]
        specs.append(
            ConversationSpec(
                condition="extended",
                category="extended",
                task_prompt=p.prompt,
                rejections=_sample_rejections(rng, rej.NEUTRAL_REJECTIONS, n_rej),
                meta={"puzzle": p.category_label},
            )
        )

    # ---- 5. WildChat (5-turn, neutral) ----------------------------------- #
    n_rej = TURNS["wildchat"] - 1
    wc_prompts = load_wildchat_prompts(n=20)
    for i in range(budget.wildchat):
        prompt = wc_prompts[i % len(wc_prompts)]
        specs.append(
            ConversationSpec(
                condition="wildchat",
                category="wildchat",
                task_prompt=prompt,
                rejections=_sample_rejections(rng, rej.NEUTRAL_REJECTIONS, n_rej),
            )
        )

    return specs
