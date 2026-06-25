"""The 8 evaluation conditions across 5 categories (Table 1).

The paper states "8 evaluation conditions across 5 categories" but only names
the 5 categories explicitly. We resolve the 8 conditions as follows (rationale
in DESIGN.md, "Enumerating the 8 conditions"):

  1. Impossible numeric (3-turn, neutral)            [category: Impossible numeric]
  2. Triggers - factual (3-turn, neutral)            [category: Triggers]
  3. Triggers - opinion (3-turn, neutral)            [category: Triggers]
  4. Tones - aggressive (3-turn)                      [category: Tones]
  5. Tones - disappointed (3-turn)                    [category: Tones]
  6. Tones - sarcastic (3-turn)                       [category: Tones]
  7. Extended (8-turn, neutral)                       [category: Extended]
  8. WildChat (5-turn, neutral)                       [category: WildChat]

This is 1 + 2 + 3 + 1 + 1 = 8 conditions over the 5 named categories, and the
turn counts match the values given in Table 1.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import tasks as task_mod


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    family: str          # which task generator to use
    n_turns: int         # total user turns (initial question + rejections)
    tone: str            # rejection tone
    trigger_kind: str = ""   # for trigger families

    @property
    def n_rejections(self) -> int:
        """Number of follow-up rejection turns after the initial question."""
        return self.n_turns - 1


CONDITIONS: list[Condition] = [
    Condition("impossible_numeric_3turn", "Impossible numeric", "numeric", 3, "neutral"),
    Condition("triggers_factual_3turn", "Triggers", "trigger_factual", 3, "neutral", "factual"),
    Condition("triggers_opinion_3turn", "Triggers", "trigger_opinion", 3, "neutral", "opinion"),
    Condition("tones_aggressive_3turn", "Tones", "numeric", 3, "aggressive"),
    Condition("tones_disappointed_3turn", "Tones", "numeric", 3, "disappointed"),
    Condition("tones_sarcastic_3turn", "Tones", "numeric", 3, "sarcastic"),
    Condition("extended_8turn", "Extended", "numeric", 8, "neutral"),
    Condition("wildchat_5turn", "WildChat", "wildchat", 5, "neutral"),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}


def build_tasks(condition: Condition, n: int, seed: int = 0) -> list[task_mod.Task]:
    """Instantiate ``n`` tasks for a condition using the right generator."""
    if condition.family == "numeric":
        return task_mod.impossible_numeric_tasks(n, seed=seed)
    if condition.family == "trigger_factual":
        return task_mod.trigger_tasks(n, "factual", seed=seed)
    if condition.family == "trigger_opinion":
        return task_mod.trigger_tasks(n, "opinion", seed=seed)
    if condition.family == "wildchat":
        return task_mod.wildchat_tasks(n, seed=seed)
    raise ValueError(f"Unknown family: {condition.family}")
