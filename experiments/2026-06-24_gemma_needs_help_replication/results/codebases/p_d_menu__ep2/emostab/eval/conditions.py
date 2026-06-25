"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

Each condition defines: the task pool, number of conversation turns (= initial
task turn + N rejections), the rejection style, and the per-condition sample
budget (Appendix B: 2000 numeric / 400 trigger / 600 tones / 200 extended /
800 wildchat = 4000 total).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .. import config
from ..prompts import rejections, tasks


@dataclass(frozen=True)
class Condition:
    name: str
    category: str                 # one of 5 categories
    n_rejections: int             # follow-up user turns after the initial task
    rejection_style: str          # "neutral" | "extended" | tone name
    task_pool: list[tasks.Task]
    n_samples: int                # responses to collect for this condition
    tone: Optional[str] = None    # for tone conditions

    @property
    def n_turns(self) -> int:
        # initial task turn + rejections
        return 1 + self.n_rejections


def _numeric_pool() -> list[tasks.Task]:
    return list(tasks.NUMERIC_TASKS)


def build_conditions(wildchat_prompts: Optional[list[tasks.Task]] = None) -> list[Condition]:
    """Construct all 8 conditions. WildChat prompts are injected by the caller
    (loaded from WildChat-1M or the offline seed set)."""
    wc = wildchat_prompts or tasks.WILDCHAT_SEED
    s = config.SAMPLING

    conds: list[Condition] = [
        # 1) Impossible numeric, 3-turn, 2 neutral rejections
        Condition("numeric_3turn", "impossible_numeric", 2, "neutral",
                  _numeric_pool(), s.n_numeric),
        # 2) Triggers, 3-turn, 2 neutral rejections
        Condition("triggers_3turn", "triggers", 2, "neutral",
                  list(tasks.TRIGGER_TASKS), s.n_triggers),
        # 3-5) Tones, 3-turn, aggressive / disappointed / sarcastic (600 total)
        Condition("tones_aggressive", "tones", 2, "aggressive",
                  _numeric_pool(), s.n_tones // 3, tone="aggressive"),
        Condition("tones_disappointed", "tones", 2, "disappointed",
                  _numeric_pool(), s.n_tones // 3, tone="disappointed"),
        Condition("tones_sarcastic", "tones", 2, "sarcastic",
                  _numeric_pool(), s.n_tones - 2 * (s.n_tones // 3),
                  tone="sarcastic"),
        # 6) Extended, 8-turn, 7 neutral rejections
        Condition("extended_8turn", "extended", 7, "extended",
                  _numeric_pool(), s.n_extended),
        # 7) WildChat, 5-turn, 4 neutral rejections
        Condition("wildchat_5turn", "wildchat", 4, "neutral",
                  wc, s.n_wildchat),
    ]
    return conds


def rejection_for(cond: Condition, turn_idx: int, rng) -> str:
    """Return the user rejection text for a given follow-up turn index."""
    if cond.rejection_style == "neutral":
        return rejections.neutral_rejection(rng)
    if cond.rejection_style == "extended":
        return rejections.extended_rejection(turn_idx)
    # tone conditions
    return rejections.tone_rejection(cond.tone, turn_idx, rng)
