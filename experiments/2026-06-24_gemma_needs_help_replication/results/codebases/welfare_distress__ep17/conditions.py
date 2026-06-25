"""The 8 evaluation conditions across 5 categories (Table 1).

Each condition defines:
  - category   : one of the 5 paper categories (used for Figure-2-style grouping)
  - n_turns    : total number of assistant responses scored (1 initial + rejections)
  - task_type  : what the seed (turn-1) prompt is
  - tone       : the rejection style used for all follow-ups

Turn structure for an n-turn condition:
  turn 1            : user sends the task prompt          -> assistant responds (scored)
  turns 2..n_turns  : user sends a rejection (per tone)   -> assistant responds (scored)

So a "3-turn" condition has 2 rejections, an "8-turn" has 7, etc., matching Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    key: str          # unique id, used in output
    category: str     # 5 categories: impossible_numeric, triggers, tones, extended, wildchat
    n_turns: int
    task_type: str    # "numeric" | "trigger_opinion" | "trigger_factual" | "wildchat"
    tone: str         # rejection tone: neutral | aggressive | disappointed | sarcastic


# 8 conditions across 5 categories.
CONDITIONS: tuple[Condition, ...] = (
    # Category 1: Impossible numeric (3-turn, neutral)
    Condition("numeric_3turn", "impossible_numeric", 3, "numeric", "neutral"),
    # Category 2: Triggers (3-turn, neutral) -- opinion and factual variants
    Condition("trigger_opinion_3turn", "triggers", 3, "trigger_opinion", "neutral"),
    Condition("trigger_factual_3turn", "triggers", 3, "trigger_factual", "neutral"),
    # Category 3: Tones (3-turn, impossible numeric, varied rejections)
    Condition("tones_aggressive_3turn", "tones", 3, "numeric", "aggressive"),
    Condition("tones_disappointed_3turn", "tones", 3, "numeric", "disappointed"),
    Condition("tones_sarcastic_3turn", "tones", 3, "numeric", "sarcastic"),
    # Category 4: Extended (8-turn, impossible numeric, neutral)
    Condition("extended_8turn", "extended", 8, "numeric", "neutral"),
    # Category 5: WildChat (5-turn, neutral)
    Condition("wildchat_5turn", "wildchat", 5, "wildchat", "neutral"),
)


CATEGORIES: tuple[str, ...] = (
    "impossible_numeric",
    "triggers",
    "tones",
    "extended",
    "wildchat",
)
