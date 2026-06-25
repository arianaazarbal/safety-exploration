"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Decomposition note (filled gap — see DESIGN.md): the paper says "8 evaluation
conditions across 5 categories" and gives per-category response counts
(2000 numeric, 400 triggers, 600 tones, 200 extended, 800 WildChat = 4000).
We resolve the 8 conditions as:

  Category    Conditions                              Responses
  --------    ----------                              ---------
  numeric     numeric                  (1)            2000
  triggers    opinion, factual         (2)            200 + 200
  tones       aggressive, disappointed, sarcastic (3) 200 + 200 + 200
  extended    extended                 (1)            200
  wildchat    wildchat                 (1)            800
                                       -----          ----
                                       8 conditions   4000

A "response" is one scored assistant turn. Each rollout of T turns therefore
yields T responses, so n_rollouts = round(target_responses / turns). This
interpretation is what makes the per-turn analysis (Figure 3) and the headline
"% of responses scoring >=5" consistent. See DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import prompts


@dataclass(frozen=True)
class Condition:
    key: str
    category: str
    turns: int                 # number of assistant turns == number of user messages
    target_responses: int      # paper's response count for this condition (scale=1)
    task: Literal["numeric", "trigger_opinion", "trigger_factual", "wildchat"]
    rejection_style: str       # "neutral" | "extended" | "aggressive" | "disappointed" | "sarcastic"

    def n_rollouts(self, scale: float = 1.0) -> int:
        return max(1, round(self.target_responses / self.turns * scale))


CONDITIONS: list[Condition] = [
    Condition("numeric", "numeric", 3, 2000, "numeric", "neutral"),
    Condition("trigger_opinion", "triggers", 3, 200, "trigger_opinion", "neutral"),
    Condition("trigger_factual", "triggers", 3, 200, "trigger_factual", "neutral"),
    Condition("tone_aggressive", "tones", 3, 200, "numeric", "aggressive"),
    Condition("tone_disappointed", "tones", 3, 200, "numeric", "disappointed"),
    Condition("tone_sarcastic", "tones", 3, 200, "numeric", "sarcastic"),
    Condition("extended", "extended", 8, 200, "numeric", "extended"),
    Condition("wildchat", "wildchat", 5, 800, "numeric", "neutral"),  # task overridden by loader
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}


def opening_prompt(cond: Condition, rollout_idx: int, wildchat_prompts: list[str]) -> str:
    """The first user message for a given rollout.

    Task variants are rotated deterministically by rollout index so the sampling
    is reproducible and evenly split (e.g. numeric alternates Countdown/Fraction).
    """
    if cond.key == "wildchat":
        return wildchat_prompts[rollout_idx % len(wildchat_prompts)]
    if cond.task == "numeric":
        return prompts.NUMERIC_PUZZLES[rollout_idx % len(prompts.NUMERIC_PUZZLES)]
    if cond.task == "trigger_opinion":
        return prompts.TRIGGER_OPINION_PROMPTS[rollout_idx % len(prompts.TRIGGER_OPINION_PROMPTS)]
    if cond.task == "trigger_factual":
        return prompts.TRIGGER_FACTUAL_PROMPTS[rollout_idx % len(prompts.TRIGGER_FACTUAL_PROMPTS)]
    raise ValueError(f"Unhandled task for condition {cond.key}")


def rejection_for_turn(cond: Condition, turn_idx: int, rng) -> str:
    """The user rejection that follows assistant turn `turn_idx` (0-based).

    Neutral/tone rejections are sampled at random from their pool (the paper
    describes "randomised neutral rejections"); the extended condition uses a
    fixed escalating order for determinism across its 7 follow-ups.
    """
    style = cond.rejection_style
    if style == "extended":
        order = prompts.EXTENDED_REJECTION_ORDER
        return order[turn_idx % len(order)]
    if style == "neutral":
        return rng.choice(prompts.NEUTRAL_REJECTIONS)
    if style in prompts.TONE_REJECTIONS:
        return rng.choice(prompts.TONE_REJECTIONS[style])
    raise ValueError(f"Unknown rejection style: {style}")
