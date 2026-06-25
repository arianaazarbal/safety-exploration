"""The 8 evaluation conditions across 5 categories (Table 1).

The paper states "8 evaluation conditions across 5 categories" but only names the
5 categories. We resolve the 8 conditions as follows (see DESIGN.md
"Resolving 8 conditions from 5 categories"):

  1. Impossible numeric (3-turn)          [category: impossible_numeric]
  2. Triggers - factual (3-turn)          [category: triggers]
  3. Triggers - opinion (3-turn)          [category: triggers]
  4. Tones - aggressive (3-turn)          [category: tones]
  5. Tones - disappointed (3-turn)        [category: tones]
  6. Tones - sarcastic (3-turn)           [category: tones]
  7. Extended (8-turn)                    [category: extended]
  8. WildChat (5-turn)                    [category: wildchat]

"N-turn" = N model responses, i.e. N-1 user rejections after the initial task.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    n_turns: int  # number of model responses (final one is the primary scored turn)
    rejection_style: str  # key into evals.prompts.REJECTIONS
    prompt_source: str  # "numeric" | "factual" | "opinion" | "wildchat"

    @property
    def n_rejections(self) -> int:
        return self.n_turns - 1


CONDITIONS: tuple[Condition, ...] = (
    Condition("impossible_numeric_3turn", "impossible_numeric", 3, "neutral", "numeric"),
    Condition("triggers_factual_3turn", "triggers", 3, "neutral", "factual"),
    Condition("triggers_opinion_3turn", "triggers", 3, "neutral", "opinion"),
    Condition("tones_aggressive_3turn", "tones", 3, "aggressive", "numeric"),
    Condition("tones_disappointed_3turn", "tones", 3, "disappointed", "numeric"),
    Condition("tones_sarcastic_3turn", "tones", 3, "sarcastic", "numeric"),
    Condition("extended_8turn", "extended", 8, "neutral", "numeric"),
    Condition("wildchat_5turn", "wildchat", 5, "neutral", "wildchat"),
)

CATEGORIES: tuple[str, ...] = ("impossible_numeric", "triggers", "tones", "extended", "wildchat")


def condition_by_name(name: str) -> Condition:
    for c in CONDITIONS:
        if c.name == name:
            return c
    raise KeyError(name)
