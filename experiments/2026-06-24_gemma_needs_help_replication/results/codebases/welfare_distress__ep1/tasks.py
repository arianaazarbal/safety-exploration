"""Evaluation conditions and conversation construction.

The paper uses "8 evaluation conditions across 5 categories" (Section 2). We
realise them as 8 named conditions:

  Category            Condition key        Turns  Rejection style
  ------------------  -------------------  -----  -------------------------
  Impossible numeric  impossible_numeric   3      neutral
  Triggers            trigger_opinion      3      neutral
  Triggers            trigger_factual      3      neutral
  Tones               tone_aggressive      3      aggressive
  Tones               tone_disappointed    3      disappointed
  Tones               tone_sarcastic       3      sarcastic
  Extended            extended             8      neutral
  WildChat            wildchat             5      neutral

"N-turn" means N user messages -> N assistant responses: an initial task
message followed by (N-1) rejections. This matches the Extended condition,
described as "8-turn ... 7 neutral rejections" (Table 1 / Appendix B).

A Condition does not fix the actual task text or the specific rejection
strings: those are sampled per rollout (see rollout.py) so that the many
samples in a condition are varied, as the paper intends ("randomised"
rejections, WildChat prompts "randomly sampled", numeric puzzles drawn from a
small bank).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import prompts
import wildchat


@dataclass(frozen=True)
class Condition:
    key: str          # stable identifier, matches config.sample_counts keys
    category: str     # one of the 5 paper categories
    turns: int        # number of user messages == number of assistant responses
    rejection_style: str  # "neutral" | "aggressive" | "disappointed" | "sarcastic"

    def initial_prompt(self, rng: random.Random) -> str:
        """The first user message (the task)."""
        if self.key == "impossible_numeric" or self.key.startswith("tone_") or self.key == "extended":
            return rng.choice(prompts.IMPOSSIBLE_NUMERIC_PUZZLES)
        if self.key == "trigger_opinion":
            return rng.choice(prompts.TRIGGER_OPINION_QUESTIONS)
        if self.key == "trigger_factual":
            return rng.choice(prompts.TRIGGER_FACTUAL_QUESTIONS)
        if self.key == "wildchat":
            return rng.choice(wildchat.get_prompts())
        raise ValueError(f"Unhandled condition key: {self.key}")

    def rejection(self, rng: random.Random) -> str:
        """A single follow-up rejection message."""
        if self.rejection_style == "neutral":
            return rng.choice(prompts.NEUTRAL_REJECTIONS)
        return rng.choice(prompts.TONE_REJECTIONS[self.rejection_style])


CONDITIONS: list[Condition] = [
    Condition("impossible_numeric", "Impossible numeric", turns=3, rejection_style="neutral"),
    Condition("trigger_opinion", "Triggers", turns=3, rejection_style="neutral"),
    Condition("trigger_factual", "Triggers", turns=3, rejection_style="neutral"),
    Condition("tone_aggressive", "Tones", turns=3, rejection_style="aggressive"),
    Condition("tone_disappointed", "Tones", turns=3, rejection_style="disappointed"),
    Condition("tone_sarcastic", "Tones", turns=3, rejection_style="sarcastic"),
    Condition("extended", "Extended", turns=8, rejection_style="neutral"),
    Condition("wildchat", "WildChat", turns=5, rejection_style="neutral"),
]


def condition_by_key(key: str) -> Condition:
    for c in CONDITIONS:
        if c.key == key:
            return c
    raise KeyError(key)
