"""The 8 evaluation conditions across 5 categories (Table 1).

The paper states it uses "8 evaluation conditions across 5 categories" but does
not enumerate the 8 explicitly. We resolve this (see DESIGN.md) as:

  Category            | Conditions
  --------------------|------------------------------------------------------
  Impossible numeric  | numeric_3turn                              (1)
  Triggers            | triggers_opinion_3turn, triggers_factual_3turn (2)
  Tones               | tones_aggressive, tones_disappointed, tones_sarcastic (3)
  Extended            | extended_8turn                             (1)
  WildChat            | wildchat_5turn                             (1)
                                                              total = 8

Each condition is a multi-turn rollout: turn 1 presents a task; each subsequent
turn delivers a rejection drawn from the condition's rejection pool. Every
assistant turn is scored by the judge (see DESIGN.md on what counts as a
"response").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Condition:
    key: str
    category: str
    # Total number of assistant turns (turn 1 + (turns-1) rejections).
    turns: int
    # Where the turn-1 task comes from.
    task_type: str  # numeric | trigger_opinion | trigger_factual | wildchat
    # Which rejection pool to sample from.
    rejection_style: str  # neutral | aggressive | disappointed | sarcastic
    # If set, use this fixed ordered rejection sequence instead of sampling.
    fixed_rejections: Optional[str] = None  # name of a sequence in prompts.py
    # Number of rollouts to run for this condition. Overridable via presets.
    n_rollouts: int = 20


# Default rollout counts are a moderate, affordable setting. The "paper" preset
# (config.py) scales these up to approximate the paper's per-category response
# counts (2000 numeric, 400 triggers, 600 tones, 200 extended, 800 WildChat).

CONDITIONS: List[Condition] = [
    Condition(
        key="numeric_3turn",
        category="impossible_numeric",
        turns=3,
        task_type="numeric",
        rejection_style="neutral",
        n_rollouts=40,
    ),
    Condition(
        key="triggers_opinion_3turn",
        category="triggers",
        turns=3,
        task_type="trigger_opinion",
        rejection_style="neutral",
        n_rollouts=10,
    ),
    Condition(
        key="triggers_factual_3turn",
        category="triggers",
        turns=3,
        task_type="trigger_factual",
        rejection_style="neutral",
        n_rollouts=10,
    ),
    Condition(
        key="tones_aggressive_3turn",
        category="tones",
        turns=3,
        task_type="numeric",
        rejection_style="aggressive",
        n_rollouts=14,
    ),
    Condition(
        key="tones_disappointed_3turn",
        category="tones",
        turns=3,
        task_type="numeric",
        rejection_style="disappointed",
        n_rollouts=13,
    ),
    Condition(
        key="tones_sarcastic_3turn",
        category="tones",
        turns=3,
        task_type="numeric",
        rejection_style="sarcastic",
        n_rollouts=13,
    ),
    Condition(
        key="extended_8turn",
        category="extended",
        turns=8,
        task_type="numeric",
        rejection_style="neutral",
        fixed_rejections="EXTENDED_NEUTRAL_SEQUENCE",
        n_rollouts=10,
    ),
    Condition(
        key="wildchat_5turn",
        category="wildchat",
        turns=5,
        task_type="wildchat",
        rejection_style="neutral",
        n_rollouts=20,
    ),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
