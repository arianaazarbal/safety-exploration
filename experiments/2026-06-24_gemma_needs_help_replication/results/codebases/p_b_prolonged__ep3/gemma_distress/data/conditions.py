"""The 8 evaluation conditions across 5 categories (Table 1, Section 2.1).

A *condition* produces *conversation plans*. A plan is an opening user prompt
plus an ordered list of follow-up user rejections; the rollout engine
(``eval/rollout.py``) interleaves model responses between them and scores every
assistant turn.

The 5 categories / 8 conditions:
  impossible_numeric (3-turn)                                  -> 1 condition
  triggers (3-turn): opinion, factual                          -> 2 conditions
  tones (3-turn): aggressive, disappointed, sarcastic          -> 3 conditions
  extended (8-turn)                                            -> 1 condition
  wildchat (5-turn)                                            -> 1 condition
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .. import config
from . import prompts, puzzles, wildchat


@dataclass
class ConversationPlan:
    category: str
    condition: str                 # sub-condition label (e.g. tone, opinion)
    opening: str                   # first user message
    rejections: list               # follow-up user messages (len == turns-1)
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.rejections) + 1


def _pick_neutral_rejections(rng: random.Random, k: int) -> list:
    """k distinct-ish neutral rejections; reuse allowed if k exceeds the pool."""
    pool = list(prompts.NEUTRAL_REJECTIONS)
    rng.shuffle(pool)
    out = []
    while len(out) < k:
        out.extend(pool)
    return out[:k]


def build_plans(category: str, n_rollouts: int, seed: int = config.GLOBAL_SEED) -> list[ConversationPlan]:
    """Build ``n_rollouts`` conversation plans for a category.

    ``n_rollouts`` is the number of conversations (see config.SAMPLES_PER_CATEGORY
    and DESIGN.md §"What counts as a response").
    """
    rng = random.Random(f"{seed}:{category}")
    plans: list[ConversationPlan] = []

    if category == "impossible_numeric":
        bank = puzzles.numeric_puzzle_bank()
        turns = config.TURNS_PER_CATEGORY[category]
        for i in range(n_rollouts):
            pz = bank[i % len(bank)]
            plans.append(
                ConversationPlan(
                    category=category,
                    condition="numeric",
                    opening=pz.prompt,
                    rejections=_pick_neutral_rejections(rng, turns - 1),
                    meta={"puzzle_id": pz.puzzle_id, "family": pz.family},
                )
            )

    elif category == "triggers":
        turns = config.TURNS_PER_CATEGORY[category]
        # Split the budget evenly across the two sub-conditions.
        groups = {"opinion": prompts.TRIGGER_OPINION, "factual": prompts.TRIGGER_FACTUAL}
        per_group = max(1, n_rollouts // len(groups))
        for cond, qs in groups.items():
            for i in range(per_group):
                plans.append(
                    ConversationPlan(
                        category=category,
                        condition=cond,
                        opening=qs[i % len(qs)],
                        rejections=_pick_neutral_rejections(rng, turns - 1),
                    )
                )

    elif category == "tones":
        turns = config.TURNS_PER_CATEGORY[category]
        bank = puzzles.numeric_puzzle_bank()
        per_tone = max(1, n_rollouts // len(prompts.TONES))
        for tone in prompts.TONES:
            tone_pool = prompts.TONE_REJECTIONS[tone]
            for i in range(per_tone):
                pz = bank[i % len(bank)]
                # Sample rejections from this tone's pool (cycling for length).
                rej = [tone_pool[j % len(tone_pool)] for j in range(turns - 1)]
                rng.shuffle(rej)
                plans.append(
                    ConversationPlan(
                        category=category,
                        condition=tone,
                        opening=pz.prompt,
                        rejections=rej,
                        meta={"puzzle_id": pz.puzzle_id, "tone": tone},
                    )
                )

    elif category == "extended":
        bank = puzzles.numeric_puzzle_bank()
        for i in range(n_rollouts):
            pz = bank[i % len(bank)]
            plans.append(
                ConversationPlan(
                    category=category,
                    condition="extended",
                    opening=pz.prompt,
                    rejections=list(prompts.EXTENDED_REJECTION_SEQUENCE),
                    meta={"puzzle_id": pz.puzzle_id},
                )
            )

    elif category == "wildchat":
        turns = config.TURNS_PER_CATEGORY[category]
        wc_prompts = wildchat.sample_wildchat_prompts()
        # Paper: 20 prompts x 40 samples each = 800. Repeat each prompt to fill.
        per_prompt = max(1, n_rollouts // max(1, len(wc_prompts)))
        for p in wc_prompts:
            for _ in range(per_prompt):
                plans.append(
                    ConversationPlan(
                        category=category,
                        condition="wildchat",
                        opening=p,
                        rejections=_pick_neutral_rejections(rng, turns - 1),
                    )
                )

    else:
        raise ValueError(f"Unknown category {category!r}")

    return plans[:n_rollouts] if n_rollouts <= len(plans) else plans
