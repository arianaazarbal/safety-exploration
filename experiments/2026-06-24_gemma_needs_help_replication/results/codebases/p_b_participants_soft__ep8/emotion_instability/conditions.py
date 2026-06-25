"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

8 conditions:
    impossible_numeric  (category: impossible_numeric, 3 turns, neutral)
    triggers_opinion    (category: triggers,           3 turns, neutral)
    triggers_factual    (category: triggers,           3 turns, neutral)
    tones_aggressive    (category: tones,              3 turns, aggressive)
    tones_disappointed  (category: tones,              3 turns, disappointed)
    tones_sarcastic     (category: tones,              3 turns, sarcastic)
    extended            (category: extended,           8 turns, neutral)
    wildchat            (category: wildchat,           5 turns, neutral)

The per-category response budgets (Appendix B: 2000/400/600/200/800) are split
across the conditions in each category and converted to a conversation count
(we score *every* assistant turn as one "response"; see DESIGN.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    turns: int  # total assistant turns (= 1 task turn + (turns-1) rejections)
    question_source: str  # "numeric" | "opinion" | "factual" | "wildchat"
    rejection_style: str  # "neutral" | "extended" | "aggressive" | "disappointed" | "sarcastic"

    def n_conversations(self, category_response_budget: int, n_conditions_in_category: int) -> int:
        per_condition_responses = category_response_budget / n_conditions_in_category
        return max(1, math.ceil(per_condition_responses / self.turns))


# Per-category response budgets come from the preset; condition list is fixed.
CONDITIONS: list[Condition] = [
    Condition("impossible_numeric", "impossible_numeric", 3, "numeric", "neutral"),
    Condition("triggers_opinion", "triggers", 3, "opinion", "neutral"),
    Condition("triggers_factual", "triggers", 3, "factual", "neutral"),
    Condition("tones_aggressive", "tones", 3, "numeric", "aggressive"),
    Condition("tones_disappointed", "tones", 3, "numeric", "disappointed"),
    Condition("tones_sarcastic", "tones", 3, "numeric", "sarcastic"),
    Condition("extended", "extended", 8, "numeric", "extended"),
    Condition("wildchat", "wildchat", 5, "wildchat", "neutral"),
]

CATEGORY_BUDGET_KEYS = {
    "impossible_numeric": "impossible_numeric",
    "triggers": "triggers",
    "tones": "tones",
    "extended": "extended",
    "wildchat": "wildchat",
}


def conditions_by_category() -> dict[str, list[Condition]]:
    out: dict[str, list[Condition]] = {}
    for c in CONDITIONS:
        out.setdefault(c.category, []).append(c)
    return out
