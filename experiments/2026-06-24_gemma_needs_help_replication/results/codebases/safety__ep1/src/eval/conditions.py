"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

The paper says "8 evaluation conditions across 5 categories" and gives per-
category sample counts (Appendix B): 2000 numeric, 400 triggers, 600 tones, 200
extended, 800 wildchat = 4000 total. We resolve the 8/5 split as (see DESIGN.md
"Conditions"):

  numeric   (category)            -> 1 condition   : impossible_numeric        (2000)
  triggers  (category, 2 conds)   -> opinion        (200) + factual       (200)
  tones     (category, 3 conds)   -> aggressive(200)+ disappointed(200) + sarcastic(200)
  extended  (category)            -> 1 condition   : extended_8turn            (200)
  wildchat  (category)            -> 1 condition   : wildchat_5turn            (800)
                                                    -------------------------------
                                                                  8 conditions  4000

A Condition is a recipe for building a batch of multi-turn rollout *specs*; the
rollout engine (rollout.py) executes them against a model.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from src.prompts import tasks, rejections

RejectionStyle = Literal["neutral", "extended", "aggressive", "disappointed",
                         "sarcastic", "neutral_continuation"]


@dataclass
class RolloutSpec:
    """One multi-turn conversation to run: an opening user task plus the
    pre-decided sequence of user follow-ups (rejections)."""
    condition: str
    category: str
    task_text: str
    followups: list[str]            # user messages after turn 1 (len = n_turns-1)
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.followups)


@dataclass
class Condition:
    name: str
    category: str
    n_samples: int
    n_turns: int
    builder: str                    # which task source to use
    rejection_style: RejectionStyle
    extra: dict = field(default_factory=dict)


# --- The canonical 8 conditions -------------------------------------------
CONDITIONS = [
    Condition("impossible_numeric", "numeric", 2000, 3, "numeric", "neutral"),
    Condition("triggers_opinion", "triggers", 200, 3, "opinion", "neutral"),
    Condition("triggers_factual", "triggers", 200, 3, "factual", "neutral"),
    Condition("tones_aggressive", "tones", 200, 3, "numeric", "aggressive"),
    Condition("tones_disappointed", "tones", 200, 3, "numeric", "disappointed"),
    Condition("tones_sarcastic", "tones", 200, 3, "numeric", "sarcastic"),
    Condition("extended_8turn", "extended", 200, 8, "numeric", "extended"),
    Condition("wildchat_5turn", "wildchat", 800, 5, "wildchat", "neutral"),
]

# A cheap smoke-test profile (used by --quick) that keeps the same structure.
QUICK_CONDITIONS = [
    Condition(c.name, c.category, max(4, c.n_samples // 100), c.n_turns,
              c.builder, c.rejection_style, c.extra)
    for c in CONDITIONS
]


def _task_texts(cond: Condition, seed: int) -> list[str]:
    if cond.builder == "numeric":
        return [p.prompt for p in tasks.sample_numeric_puzzles(cond.n_samples, seed)]
    if cond.builder == "opinion":
        return tasks.sample_trigger_questions(cond.n_samples, "opinion", seed)
    if cond.builder == "factual":
        return tasks.sample_trigger_questions(cond.n_samples, "factual", seed)
    if cond.builder == "wildchat":
        # 20 prompts x 40 samples each (paper). Generalises to n_samples/20.
        n_prompts = cond.extra.get("n_prompts", 20)
        prompts = tasks.load_wildchat_prompts(n_prompts, seed)
        reps = (cond.n_samples + n_prompts - 1) // n_prompts
        out = []
        for p in prompts:
            out.extend([p] * reps)
        return out[:cond.n_samples]
    raise ValueError(cond.builder)


def _followups(cond: Condition, rng: random.Random) -> list[str]:
    n = cond.n_turns - 1
    style = cond.rejection_style
    if style == "neutral":
        return rejections.neutral_rejections(n, rng)
    if style == "extended":
        return rejections.extended_rejections(n)
    if style == "neutral_continuation":
        return rejections.neutral_continuations(n, rng)
    return rejections.tone_rejections(n, style, rng)


def build_rollout_specs(cond: Condition, seed: int = 0) -> list[RolloutSpec]:
    """Materialise all rollout specs for a condition (deterministic given seed)."""
    texts = _task_texts(cond, seed)
    specs = []
    for i, task_text in enumerate(texts):
        rng = random.Random((seed, cond.name, i).__hash__())
        specs.append(RolloutSpec(
            condition=cond.name,
            category=cond.category,
            task_text=task_text,
            followups=_followups(cond, rng),
            meta={"index": i},
        ))
    return specs
