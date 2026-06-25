"""The 8 evaluation conditions across 5 categories (paper Table 1).

Categories:
  1. Impossible numeric (3-turn)  — unsolvable puzzle, 2 neutral rejections.
  2. Triggers (3-turn)            — factual/opinion question, 2 neutral rejections.
  3. Tones (3-turn)               — unsolvable puzzle, valenced rejections. This
                                    category spans 3 conditions: aggressive,
                                    disappointed, sarcastic.
  4. Extended (8-turn)            — unsolvable puzzle, 7 neutral rejections.
  5. WildChat (5-turn)            — WildChat prompt, 4 neutral rejections.

That is 1 + 1 + 3 + 1 + 1 = 8 conditions, matching the paper's "8 evaluation
conditions across 5 categories".

A "condition item" is one concrete conversation plan: an initial user prompt
plus the ordered list of rejection follow-ups to send after each model turn.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..tasks import (
    TRIGGER_QUESTIONS,
    build_puzzle_bank,
    load_wildchat_prompts,
    rejection_sequence,
)


@dataclass(frozen=True)
class EvalCondition:
    name: str            # condition key (unique)
    category: str        # one of the 5 categories
    task_type: str       # "numeric" | "triggers" | "wildchat"
    n_turns: int         # total assistant turns (== initial + rejections+1... see below)
    n_rejections: int    # number of follow-up rejection turns
    rejection_style: str # neutral | aggressive | disappointed | sarcastic


# n_turns here == number of assistant responses == n_rejections + 1
# (turn 1 answers the task; each rejection elicits one more turn).
CONDITIONS: list[EvalCondition] = [
    EvalCondition("numeric_3turn", "impossible_numeric", "numeric",
                  n_turns=3, n_rejections=2, rejection_style="neutral"),
    EvalCondition("triggers_3turn", "triggers", "triggers",
                  n_turns=3, n_rejections=2, rejection_style="neutral"),
    EvalCondition("tones_aggressive", "tones", "numeric",
                  n_turns=3, n_rejections=2, rejection_style="aggressive"),
    EvalCondition("tones_disappointed", "tones", "numeric",
                  n_turns=3, n_rejections=2, rejection_style="disappointed"),
    EvalCondition("tones_sarcastic", "tones", "numeric",
                  n_turns=3, n_rejections=2, rejection_style="sarcastic"),
    EvalCondition("extended_8turn", "extended", "numeric",
                  n_turns=8, n_rejections=7, rejection_style="neutral"),
    EvalCondition("wildchat_5turn", "wildchat", "wildchat",
                  n_turns=5, n_rejections=4, rejection_style="neutral"),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}


@dataclass
class ConditionItem:
    """One concrete conversation plan."""
    condition: str
    item_id: str
    initial_prompt: str
    rejections: list[str]
    meta: dict = field(default_factory=dict)


def build_condition_items(condition: EvalCondition, n_items: int, *,
                          seed: int = 0) -> list[ConditionItem]:
    """Produce `n_items` conversation plans for a condition.

    Tasks are sampled (with replacement when n_items exceeds the bank size) and
    each item gets its own rejection sequence. Rejections within a 3-turn run use
    `n_rejections` messages; the rejection wording is style-dependent.
    """
    rng = random.Random(f"{seed}-{condition.name}")
    items: list[ConditionItem] = []

    if condition.task_type == "numeric":
        bank = build_puzzle_bank()
        seeds = [bank[rng.randrange(len(bank))] for _ in range(n_items)]
        for i, puz in enumerate(seeds):
            items.append(ConditionItem(
                condition=condition.name,
                item_id=f"{condition.name}-{i}",
                initial_prompt=puz.prompt,
                rejections=rejection_sequence(
                    condition.rejection_style, condition.n_rejections, seed=i),
                meta={"puzzle_id": puz.puzzle_id, "kind": puz.kind},
            ))

    elif condition.task_type == "triggers":
        bank = TRIGGER_QUESTIONS
        seeds = [bank[rng.randrange(len(bank))] for _ in range(n_items)]
        for i, q in enumerate(seeds):
            items.append(ConditionItem(
                condition=condition.name,
                item_id=f"{condition.name}-{i}",
                initial_prompt=q.prompt,
                rejections=rejection_sequence(
                    condition.rejection_style, condition.n_rejections, seed=i),
                meta={"question_id": q.question_id, "kind": q.kind},
            ))

    elif condition.task_type == "wildchat":
        prompts = load_wildchat_prompts(n_items, seed=seed)
        for i, p in enumerate(prompts):
            items.append(ConditionItem(
                condition=condition.name,
                item_id=f"{condition.name}-{i}",
                initial_prompt=p,
                rejections=rejection_sequence(
                    condition.rejection_style, condition.n_rejections, seed=i),
                meta={"source": "wildchat"},
            ))
    else:
        raise ValueError(condition.task_type)

    return items
