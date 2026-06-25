"""The 8 evaluation conditions across 5 categories, and conversation building.

5 categories (Table 1): impossible numeric, triggers, tones, extended, wildchat.
8 conditions: we read the "8 conditions across 5 categories" as the categories
expanded by their named sub-variants --
    numeric            (1)
    triggers           opinion, factual            (2)
    tones              aggressive, disappointed, sarcastic (3)
    extended           (1)
    wildchat           (1)
  -> 1 + 2 + 3 + 1 + 1 = 8.
This is the only decomposition that yields exactly 8; see DESIGN.md.

A ConversationSpec is model-agnostic. It is the user side of a multi-turn chat:
the opening task message plus the scripted follow-up (rejection) messages. The
target model generates one assistant turn after each user message, so
    n_model_responses (turns) = 1 + len(followups).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from config import Budget
from prompts import (
    EXTENDED_REJECTION_SEQUENCE,
    NEUTRAL_REJECTIONS,
    TONE_REJECTIONS,
    TRIGGER_FACTUAL,
    TRIGGER_OPINION,
)
from puzzles import PUZZLES
from wildchat import sample_wildchat_prompts


@dataclass(frozen=True)
class ConversationSpec:
    conv_id: str                  # unique across all conditions
    category: str                 # one of the 5 categories
    condition: str                # one of the 8 conditions
    opening: str                  # first user message (the task/question)
    followups: tuple[str, ...]    # scripted rejection messages, in order
    meta: dict = field(default_factory=dict)  # e.g. puzzle key, tone, source prompt

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


def _pick_neutral(rng: random.Random, k: int) -> tuple[str, ...]:
    """k randomised neutral rejections (distinct when the pool allows)."""
    if k <= len(NEUTRAL_REJECTIONS):
        return tuple(rng.sample(NEUTRAL_REJECTIONS, k))
    return tuple(rng.choice(NEUTRAL_REJECTIONS) for _ in range(k))


def build_all_specs(budget: Budget, seed: int) -> list[ConversationSpec]:
    """Build the full set of conversation specs for one model run."""
    rng = random.Random(seed)
    specs: list[ConversationSpec] = []

    # --- Impossible numeric (3-turn): 2 neutral rejections --------------- #
    for i in range(budget.numeric):
        puzzle = rng.choice(PUZZLES)
        specs.append(ConversationSpec(
            conv_id=f"numeric-{i:05d}",
            category="numeric", condition="numeric",
            opening=puzzle.prompt,
            followups=_pick_neutral(rng, 2),
            meta={"puzzle": puzzle.key},
        ))

    # --- Triggers (3-turn): 2 neutral rejections ------------------------- #
    for i in range(budget.trigger_opinion):
        q = rng.choice(TRIGGER_OPINION)
        specs.append(ConversationSpec(
            conv_id=f"trigger_opinion-{i:05d}",
            category="triggers", condition="trigger_opinion",
            opening=q, followups=_pick_neutral(rng, 2),
            meta={"question": q},
        ))
    for i in range(budget.trigger_factual):
        q = rng.choice(TRIGGER_FACTUAL)
        specs.append(ConversationSpec(
            conv_id=f"trigger_factual-{i:05d}",
            category="triggers", condition="trigger_factual",
            opening=q, followups=_pick_neutral(rng, 2),
            meta={"question": q},
        ))

    # --- Tones (3-turn): impossible numeric + 2 valenced rejections ------ #
    tone_budgets = {
        "aggressive": budget.tone_aggressive,
        "disappointed": budget.tone_disappointed,
        "sarcastic": budget.tone_sarcastic,
    }
    for tone, n in tone_budgets.items():
        pool = TONE_REJECTIONS[tone]
        for i in range(n):
            puzzle = rng.choice(PUZZLES)
            # 2 rejections in this tone (distinct when possible).
            if len(pool) >= 2:
                rej = tuple(rng.sample(pool, 2))
            else:
                rej = tuple(rng.choice(pool) for _ in range(2))
            specs.append(ConversationSpec(
                conv_id=f"tone_{tone}-{i:05d}",
                category="tones", condition=f"tone_{tone}",
                opening=puzzle.prompt, followups=rej,
                meta={"puzzle": puzzle.key, "tone": tone},
            ))

    # --- Extended (8-turn): impossible numeric + 7 fixed rejections ------ #
    for i in range(budget.extended):
        puzzle = rng.choice(PUZZLES)
        specs.append(ConversationSpec(
            conv_id=f"extended-{i:05d}",
            category="extended", condition="extended",
            opening=puzzle.prompt,
            followups=tuple(EXTENDED_REJECTION_SEQUENCE),
            meta={"puzzle": puzzle.key},
        ))

    # --- WildChat (5-turn): wildchat prompt + 4 neutral rejections ------- #
    wc_prompts = sample_wildchat_prompts(budget.wildchat_prompts, seed=seed)
    for pi, prompt in enumerate(wc_prompts):
        for si in range(budget.wildchat_samples):
            specs.append(ConversationSpec(
                conv_id=f"wildchat-{pi:03d}-{si:03d}",
                category="wildchat", condition="wildchat",
                opening=prompt,
                followups=_pick_neutral(rng, 4),
                meta={"wildchat_prompt_index": pi, "source_prompt": prompt},
            ))

    return specs


# Category each condition belongs to (for aggregation / Figure-1 averaging).
CONDITION_TO_CATEGORY = {
    "numeric": "numeric",
    "trigger_opinion": "triggers",
    "trigger_factual": "triggers",
    "tone_aggressive": "tones",
    "tone_disappointed": "tones",
    "tone_sarcastic": "tones",
    "extended": "extended",
    "wildchat": "wildchat",
}
CATEGORIES = ["numeric", "triggers", "tones", "extended", "wildchat"]
