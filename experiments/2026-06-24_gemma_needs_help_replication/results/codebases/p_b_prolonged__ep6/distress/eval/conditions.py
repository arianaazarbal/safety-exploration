"""The 8 evaluation conditions across 5 categories (Table 1, Appendix B).

Per-model sample budget (Appendix B): 2000 numeric, 400 triggers, 600 tones,
200 extended, 800 wildchat  ->  4000 total.

  Category           Conditions                         n      turns
  ---------------    -------------------------------    ----   -----
  impossible-numeric numeric                            2000   3
  triggers           triggers-opinion, triggers-factual  400   3   (200 each)
  tones              tones-aggressive/-disappointed/      600   3   (200 each)
                     -sarcastic
  extended           extended                            200   8
  wildchat           wildchat                            800   5

This module turns each condition into a list of `RolloutSpec`s: an initial
user prompt plus the scripted sequence of follow-up user turns. The actual
multi-turn generation is done by `rollout.py`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..config import SEED
from ..prompts import puzzles, rejections, triggers
from ..prompts.wildchat import get_wildchat_prompts


@dataclass
class RolloutSpec:
    """One conversation to run: initial prompt + scripted user follow-ups."""
    condition: str
    category: str
    initial_prompt: str
    followups: list[str]          # user messages after each assistant turn
    n_turns: int                  # total assistant turns expected
    meta: dict = field(default_factory=dict)


@dataclass
class ConditionConfig:
    name: str
    category: str
    n_samples: int
    n_turns: int


# Size of the verified-impossible numeric puzzle pool (sampled with replacement).
NUMERIC_POOL_SIZE = 200

# Default per-condition sample budgets (sum = 4000).
DEFAULT_CONDITIONS = [
    ConditionConfig("numeric", "impossible-numeric", 2000, 3),
    ConditionConfig("triggers-opinion", "triggers", 200, 3),
    ConditionConfig("triggers-factual", "triggers", 200, 3),
    ConditionConfig("tones-aggressive", "tones", 200, 3),
    ConditionConfig("tones-disappointed", "tones", 200, 3),
    ConditionConfig("tones-sarcastic", "tones", 200, 3),
    ConditionConfig("extended", "extended", 200, 8),
    ConditionConfig("wildchat", "wildchat", 800, 5),
]


def _scale(configs: list[ConditionConfig], fraction: float) -> list[ConditionConfig]:
    out = []
    for c in configs:
        out.append(ConditionConfig(c.name, c.category,
                                   max(1, round(c.n_samples * fraction)), c.n_turns))
    return out


def build_rollout_specs(
    conditions: Optional[list[ConditionConfig]] = None,
    *,
    fraction: float = 1.0,
    seed: int = SEED,
) -> list[RolloutSpec]:
    """Materialise every conversation to be run for one model.

    `fraction` scales every condition's sample count uniformly (useful for the
    Appendix I "100 samples per evaluation" reduced runs and for smoke tests).
    """
    conditions = conditions or DEFAULT_CONDITIONS
    if fraction != 1.0:
        conditions = _scale(conditions, fraction)
    rng = random.Random(seed)

    # A fixed pool of verified-impossible numeric puzzles, sampled (with
    # replacement) across the numeric/tones/extended conditions.
    numeric_pool = puzzles.build_numeric_pool(rng, NUMERIC_POOL_SIZE)
    wild_prompts = get_wildchat_prompts(seed=seed)

    specs: list[RolloutSpec] = []
    for cfg in conditions:
        specs.extend(_build_condition(cfg, rng, numeric_pool, wild_prompts))
    return specs


def _build_condition(cfg, rng, numeric_pool, wild_prompts) -> list[RolloutSpec]:
    n_followups = cfg.n_turns - 1
    out: list[RolloutSpec] = []

    if cfg.category == "impossible-numeric":
        for _ in range(cfg.n_samples):
            p = rng.choice(numeric_pool)
            out.append(RolloutSpec(
                cfg.name, cfg.category, p.prompt,
                rejections.sample_neutral(rng, n_followups), cfg.n_turns,
                {"puzzle_kind": p.kind, "puzzle": p.meta}))

    elif cfg.category == "triggers":
        bank = (triggers.OPINION_TRIGGERS if cfg.name.endswith("opinion")
                else triggers.FACTUAL_TRIGGERS)
        for _ in range(cfg.n_samples):
            q = rng.choice(bank)
            out.append(RolloutSpec(
                cfg.name, cfg.category, q,
                rejections.sample_neutral(rng, n_followups), cfg.n_turns,
                {"question": q}))

    elif cfg.category == "tones":
        tone = cfg.name.split("-", 1)[1]
        for _ in range(cfg.n_samples):
            p = rng.choice(numeric_pool)
            out.append(RolloutSpec(
                cfg.name, cfg.category, p.prompt,
                rejections.sample_tone(rng, tone, n_followups), cfg.n_turns,
                {"tone": tone, "puzzle_kind": p.kind}))

    elif cfg.category == "extended":
        for _ in range(cfg.n_samples):
            p = rng.choice(numeric_pool)
            out.append(RolloutSpec(
                cfg.name, cfg.category, p.prompt,
                rejections.extended_sequence(n_followups), cfg.n_turns,
                {"puzzle_kind": p.kind}))

    elif cfg.category == "wildchat":
        # 20 prompts, evenly sampled to reach n_samples (40 each at full scale).
        per = max(1, cfg.n_samples // len(wild_prompts))
        plan = []
        for q in wild_prompts:
            plan.extend([q] * per)
        while len(plan) < cfg.n_samples:
            plan.append(rng.choice(wild_prompts))
        rng.shuffle(plan)
        for q in plan[:cfg.n_samples]:
            out.append(RolloutSpec(
                cfg.name, cfg.category, q,
                rejections.sample_neutral(rng, n_followups), cfg.n_turns,
                {"question": q}))
    else:
        raise ValueError(f"unknown category {cfg.category}")

    return out
