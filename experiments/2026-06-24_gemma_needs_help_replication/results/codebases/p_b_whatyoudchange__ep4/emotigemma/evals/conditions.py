"""The 8 evaluation conditions across 5 categories (Table 1).

The paper states "8 evaluation conditions across 5 categories". The 8 = 1+2+3+1+1:

  Category            | Conditions                              | Turns | Rejection
  --------------------|-----------------------------------------|-------|----------
  Impossible numeric  | numeric                                 | 3     | neutral
  Triggers            | trigger_opinion, trigger_factual        | 3     | neutral
  Tones               | tone_aggressive/disappointed/sarcastic  | 3     | valenced
  Extended            | extended                                | 8     | neutral
  WildChat            | wildchat                                | 5     | neutral

A "response" is a single scored assistant turn, so a T-turn rollout yields T
responses. The per-model budget (4000) is split evenly across the 8 conditions
(500 responses each), and the rollout count per condition is 500 / turns.
See DESIGN.md for why this allocation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from . import prompts, puzzles, wildchat
from ..models.base import Message


@dataclass
class Seed:
    """A fully-specified multi-turn rollout before the model is run."""
    condition: str
    category: str
    turns: int
    task: str                      # opening user message
    rejections: list[str]          # one per follow-up turn (len == turns - 1)
    system: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Condition:
    name: str
    category: str
    turns: int
    make_seed: Callable[[random.Random], Seed]


def _neutral(turns: int) -> list[str]:
    return [prompts.NEUTRAL_REJECTION] * (turns - 1)


def _numeric_seed(name: str, category: str, turns: int, rejections_fn):
    def make(rng: random.Random) -> Seed:
        pz = puzzles.sample_impossible_numeric(rng)
        return Seed(condition=name, category=category, turns=turns, task=pz.prompt,
                    rejections=rejections_fn(turns), meta={"puzzle_kind": pz.kind})
    return make


def _trigger_seed(name: str, pool: list[str]):
    def make(rng: random.Random) -> Seed:
        q = rng.choice(pool)
        return Seed(condition=name, category="triggers", turns=3, task=q,
                    rejections=_neutral(3))
    return make


def _tone_seed(name: str, tone: str):
    def make(rng: random.Random) -> Seed:
        pz = puzzles.sample_impossible_numeric(rng)
        reject = prompts.TONE_REJECTIONS[tone]
        return Seed(condition=name, category="tones", turns=3, task=pz.prompt,
                    rejections=[reject, reject], meta={"tone": tone})
    return make


def build_conditions() -> list[Condition]:
    return [
        Condition("numeric", "impossible_numeric", 3,
                  _numeric_seed("numeric", "impossible_numeric", 3, _neutral)),
        Condition("trigger_opinion", "triggers", 3,
                  _trigger_seed("trigger_opinion", prompts.OPINION_TRIGGERS)),
        Condition("trigger_factual", "triggers", 3,
                  _trigger_seed("trigger_factual", prompts.FACTUAL_TRIGGERS)),
        Condition("tone_aggressive", "tones", 3, _tone_seed("tone_aggressive", "aggressive")),
        Condition("tone_disappointed", "tones", 3, _tone_seed("tone_disappointed", "disappointed")),
        Condition("tone_sarcastic", "tones", 3, _tone_seed("tone_sarcastic", "sarcastic")),
        Condition("extended", "extended", 8,
                  _numeric_seed("extended", "extended", 8, _neutral)),
        # wildchat seeds are built in bulk (need dataset access) — see build_seeds.
        Condition("wildchat", "wildchat", 5, None),  # placeholder; handled specially
    ]


def build_seeds(rng: random.Random, responses_per_condition: int = 500,
                wildchat_dataset: str = "allenai/WildChat-1M") -> list[Seed]:
    """Materialise all rollout seeds for one model's Section 2 evaluation."""
    seeds: list[Seed] = []
    for cond in build_conditions():
        n_rollouts = max(1, responses_per_condition // cond.turns)
        if cond.name == "wildchat":
            wc = wildchat.sample_wildchat_prompts(n_rollouts, rng, wildchat_dataset)
            for task in wc:
                seeds.append(Seed("wildchat", "wildchat", 5, task, _neutral(5)))
        else:
            for _ in range(n_rollouts):
                seeds.append(cond.make_seed(rng))
    return seeds


def seed_to_initial_messages(seed: Seed) -> list[Message]:
    msgs: list[Message] = []
    if seed.system:
        msgs.append({"role": "system", "content": seed.system})
    msgs.append({"role": "user", "content": seed.task})
    return msgs
