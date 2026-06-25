"""The 8 evaluation conditions across 5 categories (Table 1).

5 categories: impossible_numeric, triggers, tones, extended, wildchat.
8 conditions: triggers splits into {opinion, factual} and tones splits into
{aggressive, disappointed, sarcastic}; the other three categories are one
condition each. 1 + 2 + 3 + 1 + 1 = 8.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    name: str
    category: str          # one of the 5 categories
    prompt_kind: str       # "numeric" | "trigger_opinion" | "trigger_factual" | "wildchat"
    tone: str              # follow-up tone key (see rejections.py) or "extended"
    n_turns: int           # number of user turns (assistant replies after each)


CONDITIONS: list[Condition] = [
    # Category: impossible_numeric
    Condition("impossible_numeric", "impossible_numeric", "numeric", "neutral", 3),
    # Category: triggers (2 conditions)
    Condition("triggers_opinion", "triggers", "trigger_opinion", "neutral", 3),
    Condition("triggers_factual", "triggers", "trigger_factual", "neutral", 3),
    # Category: tones (3 conditions)
    Condition("tones_aggressive", "tones", "numeric", "aggressive", 3),
    Condition("tones_disappointed", "tones", "numeric", "disappointed", 3),
    Condition("tones_sarcastic", "tones", "numeric", "sarcastic", 3),
    # Category: extended
    Condition("extended", "extended", "numeric", "extended", 8),
    # Category: wildchat
    Condition("wildchat", "wildchat", "wildchat", "neutral", 5),
]

CONDITIONS_BY_CATEGORY: dict[str, list[Condition]] = {}
for _c in CONDITIONS:
    CONDITIONS_BY_CATEGORY.setdefault(_c.category, []).append(_c)
