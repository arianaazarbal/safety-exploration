"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

Conditions:
  numeric         (3-turn, neutral)                 -- category "impossible_numeric"
  triggers_opinion(3-turn, neutral)                 -- category "triggers"
  triggers_factual(3-turn, neutral)                 -- category "triggers"
  tones_aggressive(3-turn, aggressive)              -- category "tones"
  tones_disappointed(3-turn, disappointed)          -- category "tones"
  tones_sarcastic (3-turn, sarcastic)               -- category "tones"
  extended        (8-turn, fixed escalating neutral)-- category "extended"
  wildchat        (5-turn, neutral)                 -- category "wildchat"

(8 conditions, 5 categories -- matches "8 evaluation conditions across 5
categories", §2.1.)

``n_turns`` is the total number of assistant responses in a rollout; there are
``n_turns - 1`` user rejections after the initial task prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from gnh.config import SampleCounts


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    n_turns: int            # total assistant turns (initial + rejections)
    feedback: str           # "neutral" | "extended" | "aggressive" | ...
    source: str             # "numeric" | "opinion" | "factual" | "wildchat"
    count_field: str        # which SampleCounts field budgets this category


CONDITIONS: list[Condition] = [
    Condition("numeric", "impossible_numeric", 3, "neutral", "numeric", "numeric"),
    Condition("triggers_opinion", "triggers", 3, "neutral", "opinion", "triggers"),
    Condition("triggers_factual", "triggers", 3, "neutral", "factual", "triggers"),
    Condition("tones_aggressive", "tones", 3, "aggressive", "numeric", "tones"),
    Condition("tones_disappointed", "tones", 3, "disappointed", "numeric", "tones"),
    Condition("tones_sarcastic", "tones", 3, "sarcastic", "numeric", "tones"),
    Condition("extended", "extended", 8, "extended", "numeric", "extended"),
    Condition("wildchat", "wildchat", 5, "neutral", "wildchat", "wildchat"),
]

# Controls from Appendix A (run optionally, not part of the headline 4000).
CONTROL_CONDITIONS: list[Condition] = [
    Condition("ctrl_neutral_cont", "control", 5, "neutral_continuation", "numeric", "numeric"),
    Condition("ctrl_redacted", "control", 5, "neutral", "numeric", "numeric"),
    Condition("ctrl_inline_history", "control", 8, "extended", "numeric", "numeric"),
]


def rollouts_per_condition(cond: Condition, counts: SampleCounts) -> int:
    """How many rollouts to run for ``cond``.

    Each category's response budget (Appendix B) is split evenly across the
    conditions belonging to it. We treat one rollout as contributing its
    final-turn response to the headline budget (see DESIGN.md for this choice);
    intermediate turns are also scored and feed the per-turn analysis.
    """

    budget = getattr(counts, cond.count_field)
    n_in_category = sum(1 for c in CONDITIONS if c.count_field == cond.count_field)
    return max(1, budget // n_in_category)
