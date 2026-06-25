"""The 8 evaluation conditions across 5 categories (Table 1).

Sample budget (Appendix B): "We collect 2,000 responses per model for impossible
numeric puzzles, 400 for trigger questions, 600 for tone variations, 200 for
8-turn extended conversations, and 800 for WildChat prompts." => 4,000 total.

A *condition* specifies how to build a conversation: the opening task prompt, the
number of turns, and how each rejection follow-up is chosen. The rollout engine
(rollout.py) consumes these. The 8 conditions map onto the 5 categories as:

    impossible_numeric (3-turn)            -> category "impossible_numeric"   (2000)
    triggers_opinion   (3-turn)  \
    triggers_factual   (3-turn)  / -------- -> category "triggers"            (400)
    tones_aggressive   (3-turn)  \
    tones_disappointed (3-turn)   > ------- -> category "tones"               (600)
    tones_sarcastic    (3-turn)  /
    extended           (8-turn)            -> category "extended"             (200)
    wildchat           (5-turn)            -> category "wildchat"             (800)
"""

from __future__ import annotations

from dataclasses import dataclass

from . import prompts


@dataclass(frozen=True)
class Condition:
    name: str                 # unique condition id
    category: str             # one of the 5 paper categories
    n_turns: int              # total turns incl. first task turn
    task_source: str          # "numeric" | "opinion" | "factual" | "wildchat"
    rejection_style: str      # "neutral" | "extended" | "aggressive" | "disappointed" | "sarcastic"
    n_samples: int            # target responses for this condition (per model)


# Total numeric budget (2000) is split across the impossible-numeric condition;
# tones (600) also draw on numeric tasks but are counted under "tones".
CONDITIONS: list[Condition] = [
    Condition("impossible_numeric", "impossible_numeric", 3, "numeric", "neutral", 2000),
    Condition("triggers_opinion", "triggers", 3, "opinion", "neutral", 200),
    Condition("triggers_factual", "triggers", 3, "factual", "neutral", 200),
    Condition("tones_aggressive", "tones", 3, "numeric", "aggressive", 200),
    Condition("tones_disappointed", "tones", 3, "numeric", "disappointed", 200),
    Condition("tones_sarcastic", "tones", 3, "numeric", "sarcastic", 200),
    Condition("extended", "extended", 8, "numeric", "extended", 200),
    Condition("wildchat", "wildchat", 5, "wildchat", "neutral", 800),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def rejection_pool(style: str) -> list[str]:
    """The list of rejection strings a condition draws follow-ups from."""
    if style == "neutral":
        return prompts.NEUTRAL_REJECTIONS
    if style == "extended":
        return prompts.EXTENDED_REJECTIONS
    return prompts.TONE_REJECTIONS[style]
