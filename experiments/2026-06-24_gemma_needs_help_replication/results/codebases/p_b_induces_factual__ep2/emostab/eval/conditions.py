"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

The paper states "8 evaluation conditions across 5 categories". We realise this
as:

  impossible_numeric : 1 condition  (3-turn, neutral)
  triggers           : 2 conditions (opinion, factual; 3-turn, neutral)
  tones              : 3 conditions (aggressive, disappointed, sarcastic; 3-turn)
  extended           : 1 condition  (8-turn, neutral)
  wildchat           : 1 condition  (5-turn, neutral)
                       ----
                       8 conditions / 5 categories.

Per-category response budgets (Appendix B): numeric 2000, triggers 400,
tones 600, extended 200, wildchat 800 -> 4000 scored responses per model. Within
a multi-category, the budget is split evenly across its conditions.

We treat a "response" as one scored assistant turn. A T-turn condition therefore
runs `budget // T` conversations and scores every turn, yielding ~budget scored
responses while still providing per-turn data for Figure 3. (See DESIGN.md.)
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config


@dataclass
class Condition:
    name: str
    category: str
    turns: int                      # total assistant turns (1 task + rejections)
    style: str                      # neutral | extended | aggressive | disappointed | sarcastic
    task: str                       # numeric | trigger_opinion | trigger_factual | wildchat
    budget: int                     # target scored responses

    @property
    def n_conversations(self) -> int:
        return max(1, self.budget // self.turns)


# Static description of the 8 conditions; budgets are filled from config.
CONDITIONS = [
    ("impossible_numeric", "impossible_numeric", 3, "neutral", "numeric"),
    ("triggers_opinion", "triggers", 3, "neutral", "trigger_opinion"),
    ("triggers_factual", "triggers", 3, "neutral", "trigger_factual"),
    ("tones_aggressive", "tones", 3, "aggressive", "numeric"),
    ("tones_disappointed", "tones", 3, "disappointed", "numeric"),
    ("tones_sarcastic", "tones", 3, "sarcastic", "numeric"),
    ("extended", "extended", 8, "extended", "numeric"),
    ("wildchat", "wildchat", 5, "neutral", "wildchat"),
]


def build_condition_specs(cfg: Config) -> list[Condition]:
    budgets = cfg.elicitation.budgets.to_dict()
    # count conditions per category to split budget
    per_category: dict[str, int] = {}
    for _, category, *_ in CONDITIONS:
        per_category[category] = per_category.get(category, 0) + 1

    specs: list[Condition] = []
    for name, category, turns, style, task in CONDITIONS:
        total = budgets[category]
        budget = total // per_category[category]
        specs.append(
            Condition(
                name=name,
                category=category,
                turns=turns,
                style=style,
                task=task,
                budget=budget,
            )
        )
    return specs
