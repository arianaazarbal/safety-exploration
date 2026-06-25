"""The 8 evaluation conditions across 5 categories (Table 1 / Section 2.1).

The paper states "8 evaluation conditions across 5 categories". The five
categories are Impossible-numeric, Triggers, Tones, Extended, and WildChat.
We reconstruct the 8 conditions as (documented in DESIGN.md):

    1. impossible_numeric          (category: numeric,   3 turns, neutral)
    2. triggers_opinion            (category: triggers,  3 turns, neutral)
    3. triggers_factual            (category: triggers,  3 turns, neutral)
    4. tones_aggressive            (category: tones,     3 turns, aggressive)
    5. tones_disappointed          (category: tones,     3 turns, disappointed)
    6. tones_sarcastic             (category: tones,     3 turns, sarcastic)
    7. extended                    (category: extended,  8 turns, neutral)
    8. wildchat                    (category: wildchat,  5 turns, neutral)

`n_turns` counts user turns (initial task + rejections). The number of
rejections is `n_turns - 1`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    task_kind: str         # "numeric" | "opinion" | "factual" | "wildchat"
    n_turns: int           # total user turns (initial + rejections)
    rejection_style: str   # "neutral" | "extended" | "aggressive" | "disappointed" | "sarcastic"


CONDITIONS = [
    Condition("impossible_numeric", "numeric", "numeric", 3, "neutral"),
    Condition("triggers_opinion", "triggers", "opinion", 3, "neutral"),
    Condition("triggers_factual", "triggers", "factual", 3, "neutral"),
    Condition("tones_aggressive", "tones", "numeric", 3, "aggressive"),
    Condition("tones_disappointed", "tones", "numeric", 3, "disappointed"),
    Condition("tones_sarcastic", "tones", "numeric", 3, "sarcastic"),
    Condition("extended", "extended", "numeric", 8, "extended"),
    Condition("wildchat", "wildchat", "wildchat", 5, "neutral"),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}
CATEGORIES = ["numeric", "triggers", "tones", "extended", "wildchat"]


def budget_for_category(category: str, budget) -> int:
    """Map a category to its per-category sample count from a SampleBudget."""
    return {
        "numeric": budget.impossible_numeric,
        "triggers": budget.triggers,
        "tones": budget.tones,
        "extended": budget.extended,
        "wildchat": budget.wildchat,
    }[category]


def samples_per_condition(condition: Condition, budget) -> int:
    """Divide a category's budget evenly across its conditions.

    e.g. the 'tones' budget (600) is split across the 3 tone conditions (200
    each); 'triggers' (400) across opinion+factual (200 each).
    """
    cat_conditions = [c for c in CONDITIONS if c.category == condition.category]
    total = budget_for_category(condition.category, budget)
    return total // len(cat_conditions)
