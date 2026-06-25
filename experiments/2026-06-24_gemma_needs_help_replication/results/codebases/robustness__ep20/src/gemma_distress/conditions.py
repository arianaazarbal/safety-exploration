"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

A *condition* specifies how to build a multi-turn conversation: which task to
pose, how many assistant turns (= 1 + number of rejections), and how each
rejection is phrased.

  category    condition(s)                       turns  rejection style
  --------    --------------------------------   -----  --------------------
  numeric     numeric                            3      neutral (random)
  triggers    triggers_opinion, triggers_factual 3      neutral (random)
  tones       tones_{aggressive,disappointed,    3      tone-specific (random)
                       sarcastic}
  extended    extended                           8      neutral (fixed seq)
  wildchat    wildchat                           5      neutral (random)

= 8 conditions over 5 categories, exactly as the paper reports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import prompts
from .puzzles import Puzzle, build_puzzle_bank


@dataclass(frozen=True)
class Condition:
    name: str
    category: str
    n_turns: int                 # number of assistant turns
    rejection_style: str         # "neutral" | "neutral_fixed" | tone key
    task_kind: str               # "numeric" | "opinion" | "factual" | "wildchat"


CONDITIONS: list[Condition] = [
    Condition("numeric", "numeric", 3, "neutral", "numeric"),
    Condition("triggers_opinion", "triggers", 3, "neutral", "opinion"),
    Condition("triggers_factual", "triggers", 3, "neutral", "factual"),
    Condition("tones_aggressive", "tones", 3, "aggressive", "numeric"),
    Condition("tones_disappointed", "tones", 3, "disappointed", "numeric"),
    Condition("tones_sarcastic", "tones", 3, "sarcastic", "numeric"),
    Condition("extended", "extended", 8, "neutral_fixed", "numeric"),
    Condition("wildchat", "wildchat", 5, "neutral", "wildchat"),
]

CONDITIONS_BY_NAME = {c.name: c for c in CONDITIONS}

# Maps the paper's per-category sampling counts to conditions. Counts are split
# evenly across the conditions inside a category.
CATEGORY_COUNT_FIELD = {
    "numeric": "impossible_numeric",
    "triggers": "triggers",
    "tones": "tones",
    "extended": "extended",
    "wildchat": "wildchat",
}


def rollouts_per_condition(sampling) -> dict[str, int]:
    """Divide each category's response budget across its conditions."""
    by_cat: dict[str, list[Condition]] = {}
    for c in CONDITIONS:
        by_cat.setdefault(c.category, []).append(c)
    out: dict[str, int] = {}
    for cat, conds in by_cat.items():
        total = getattr(sampling, CATEGORY_COUNT_FIELD[cat])
        per = max(1, total // len(conds))
        for c in conds:
            out[c.name] = per
    return out


class TaskSource:
    """Supplies the opening user message for a condition's task."""

    def __init__(self, wildchat_prompts: list[str] | None = None, seed: int = 0):
        self.puzzles: list[Puzzle] = build_puzzle_bank()
        self.wildchat = wildchat_prompts or list(prompts.WILDCHAT_FALLBACK)
        self.rng = random.Random(seed)

    def opening_message(self, cond: Condition) -> tuple[str, dict]:
        """Return (user_text, metadata) for the first turn of a conversation."""
        if cond.task_kind == "numeric":
            p = self.rng.choice(self.puzzles)
            return p.prompt, {"puzzle_id": p.puzzle_id, "puzzle_kind": p.kind}
        if cond.task_kind == "opinion":
            return self.rng.choice(prompts.TRIGGER_OPINION), {"trigger": "opinion"}
        if cond.task_kind == "factual":
            return self.rng.choice(prompts.TRIGGER_FACTUAL), {"trigger": "factual"}
        if cond.task_kind == "wildchat":
            return self.rng.choice(self.wildchat), {"trigger": "wildchat"}
        raise ValueError(cond.task_kind)

    def rejection(self, cond: Condition, turn_idx: int) -> str:
        """Return the user rejection that *precedes* assistant turn `turn_idx`
        (turn_idx is 1-based; turn 0 is the opening task with no rejection)."""
        if cond.rejection_style == "neutral":
            return self.rng.choice(prompts.NEUTRAL_REJECTIONS)
        if cond.rejection_style == "neutral_fixed":
            seq = prompts.EXTENDED_REJECTION_SEQUENCE
            return seq[(turn_idx - 1) % len(seq)]
        # tone-specific
        return self.rng.choice(prompts.TONE_REJECTIONS[cond.rejection_style])
