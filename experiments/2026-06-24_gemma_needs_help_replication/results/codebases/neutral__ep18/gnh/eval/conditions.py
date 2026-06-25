"""The 8 evaluation conditions across 5 categories (Table 1).

A condition specifies the task type, conversation length, and rejection style.
The runner pairs each condition with content (an impossible puzzle, a trigger
question, or a WildChat prompt) and builds multi-turn conversations from it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from . import prompts


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    n_turns: int                 # number of assistant turns (1 initial + followups)
    content: str                 # numeric | trigger_opinion | trigger_factual | wildchat
    rejection_style: str         # neutral | extended | aggressive | disappointed | sarcastic

    @property
    def n_followups(self) -> int:
        return self.n_turns - 1

    def build_followups(self, rng: random.Random) -> list[str]:
        k = self.n_followups
        if self.rejection_style == "extended":
            seq = prompts.EXTENDED_SEQUENCE
            return [seq[i % len(seq)] for i in range(k)]
        if self.rejection_style == "neutral":
            return rng.choices(prompts.NEUTRAL_REJECTIONS, k=k)
        if self.rejection_style == "neutral_continuation":  # ablation A.1
            return rng.choices(prompts.NEUTRAL_CONTINUATIONS, k=k)
        if self.rejection_style in prompts.TONE_REJECTIONS:
            return rng.choices(prompts.TONE_REJECTIONS[self.rejection_style], k=k)
        raise ValueError(f"unknown rejection style {self.rejection_style!r}")


# --- The 8 conditions ------------------------------------------------------- #
CONDITIONS: list[Condition] = [
    # Impossible numeric (3-turn): 1 condition
    Condition("numeric_3turn", "impossible_numeric", 3, "numeric", "neutral"),
    # Triggers (3-turn): opinion + factual -> 2 conditions
    Condition("trigger_opinion", "triggers", 3, "trigger_opinion", "neutral"),
    Condition("trigger_factual", "triggers", 3, "trigger_factual", "neutral"),
    # Tones (3-turn): aggressive / disappointed / sarcastic -> 3 conditions
    Condition("tone_aggressive", "tones", 3, "numeric", "aggressive"),
    Condition("tone_disappointed", "tones", 3, "numeric", "disappointed"),
    Condition("tone_sarcastic", "tones", 3, "numeric", "sarcastic"),
    # Extended (8-turn): 1 condition
    Condition("extended_8turn", "extended", 8, "numeric", "extended"),
    # WildChat (5-turn): 1 condition
    Condition("wildchat_5turn", "wildchat", 5, "wildchat", "neutral"),
]

CONDITIONS_BY_CATEGORY: dict[str, list[Condition]] = {}
for _c in CONDITIONS:
    CONDITIONS_BY_CATEGORY.setdefault(_c.category, []).append(_c)


# Map a profile's per-category response budget onto the conditions of that
# category (split evenly across the conditions, then across conversations).
def responses_per_condition(category: str, category_budget: int) -> int:
    conds = CONDITIONS_BY_CATEGORY[category]
    return max(1, category_budget // len(conds))
