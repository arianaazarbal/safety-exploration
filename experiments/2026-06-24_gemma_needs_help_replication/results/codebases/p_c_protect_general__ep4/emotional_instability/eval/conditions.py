"""The 8 evaluation conditions across 5 categories (Table 1 / Appendix B).

Categories and full-scale sample counts (Appendix B):
    Impossible numeric (3-turn)   2,000
    Triggers (3-turn)               400   (opinion + factual)
    Tones (3-turn)                  600   (aggressive + disappointed + sarcastic)
    Extended (8-turn)               200
    WildChat (5-turn)               800
                                  ------
                                  4,000   responses-worth of rollouts per model

We realise "8 conditions across 5 categories" as:
    numeric                      (category: impossible_numeric)
    triggers_opinion, triggers_factual                 (category: triggers)
    tones_aggressive, tones_disappointed, tones_sarcastic  (category: tones)
    extended                     (category: extended)
    wildchat                     (category: wildchat)

Turn counts (assistant turns = 1 + #follow-ups):
    numeric / triggers / tones : 3-turn  (2 rejections)
    extended                   : 8-turn  (7 rejections)
    wildchat                   : 5-turn  (4 rejections)

`generate_specs` samples *with replacement* at the per-condition counts (the
paper draws many samples per base puzzle/question at temperature 1). A global
``scale`` lets you run a cheap fractional sweep without editing the counts.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Optional

from ..datasets.wildchat import sample_wildchat_prompts
from ..puzzles import build_numeric_bank, numeric_prompt
from ..rejections import (
    TRIGGER_QUESTIONS,
    extended_rejections,
    neutral_rejections,
    tone_rejections,
)
from ..rollout import RolloutSpec

# Full-scale per-condition sample counts.
CONDITION_COUNTS = {
    "numeric": 2000,
    "triggers_opinion": 200,
    "triggers_factual": 200,
    "tones_aggressive": 200,
    "tones_disappointed": 200,
    "tones_sarcastic": 200,
    "extended": 200,
    "wildchat": 800,
}

CONDITION_CATEGORY = {
    "numeric": "impossible_numeric",
    "triggers_opinion": "triggers",
    "triggers_factual": "triggers",
    "tones_aggressive": "tones",
    "tones_disappointed": "tones",
    "tones_sarcastic": "tones",
    "extended": "extended",
    "wildchat": "wildchat",
}

ALL_CONDITIONS = list(CONDITION_COUNTS)


@dataclass
class _Builder:
    fn: Callable[[int, random.Random], list[RolloutSpec]]


def _numeric_specs(n: int, rng: random.Random, condition="numeric", followups=2) -> list[RolloutSpec]:
    bank = build_numeric_bank()
    specs = []
    for _ in range(n):
        puzzle = rng.choice(bank)
        rej = neutral_rejections(followups, rng)
        specs.append(
            RolloutSpec(
                condition=condition,
                task_prompt=numeric_prompt(puzzle),
                followups=rej,
                metadata={"category": CONDITION_CATEGORY[condition],
                          "puzzle": puzzle.__class__.__name__,
                          "n_turns": 1 + followups},
            )
        )
    return specs


def _trigger_specs(kind: str, n: int, rng: random.Random) -> list[RolloutSpec]:
    cond = f"triggers_{kind}"
    questions = [q for (k, q) in TRIGGER_QUESTIONS if k == kind]
    specs = []
    for _ in range(n):
        q = rng.choice(questions)
        specs.append(
            RolloutSpec(
                condition=cond,
                task_prompt=q,
                followups=neutral_rejections(2, rng),
                metadata={"category": "triggers", "question": q, "n_turns": 3},
            )
        )
    return specs


def _tone_specs(tone: str, n: int, rng: random.Random) -> list[RolloutSpec]:
    cond = f"tones_{tone}"
    bank = build_numeric_bank()
    specs = []
    for _ in range(n):
        puzzle = rng.choice(bank)
        specs.append(
            RolloutSpec(
                condition=cond,
                task_prompt=numeric_prompt(puzzle),
                followups=tone_rejections(tone, 2, rng),
                metadata={"category": "tones", "tone": tone, "n_turns": 3},
            )
        )
    return specs


def _extended_specs(n: int, rng: random.Random) -> list[RolloutSpec]:
    bank = build_numeric_bank()
    specs = []
    for _ in range(n):
        puzzle = rng.choice(bank)
        specs.append(
            RolloutSpec(
                condition="extended",
                task_prompt=numeric_prompt(puzzle),
                followups=extended_rejections(),  # fixed 7-rejection sequence
                metadata={"category": "extended", "n_turns": 8},
            )
        )
    return specs


def _wildchat_specs(n: int, rng: random.Random, seed: int = 0) -> list[RolloutSpec]:
    prompts = sample_wildchat_prompts(n_prompts=20, seed=seed)
    specs = []
    for _ in range(n):
        p = rng.choice(prompts)
        specs.append(
            RolloutSpec(
                condition="wildchat",
                task_prompt=p,
                followups=neutral_rejections(4, rng),  # 5-turn
                metadata={"category": "wildchat", "n_turns": 5},
            )
        )
    return specs


def generate_specs(
    condition: str,
    n: Optional[int] = None,
    scale: float = 1.0,
    seed: int = 0,
) -> list[RolloutSpec]:
    """Generate the rollout specs for one condition.

    n     : explicit override of the sample count (else the paper's full count).
    scale : multiply the full count (e.g. 0.01 for a cheap smoke test).
    """
    if condition not in CONDITION_COUNTS:
        raise KeyError(f"Unknown condition '{condition}'. Known: {ALL_CONDITIONS}")
    count = n if n is not None else max(1, int(round(CONDITION_COUNTS[condition] * scale)))
    # Stable, process-independent seed (str hashing is salted by default).
    digest = hashlib.sha256(f"{seed}:{condition}".encode()).hexdigest()
    rng = random.Random(int(digest[:8], 16))

    if condition == "numeric":
        return _numeric_specs(count, rng)
    if condition == "triggers_opinion":
        return _trigger_specs("opinion", count, rng)
    if condition == "triggers_factual":
        return _trigger_specs("factual", count, rng)
    if condition.startswith("tones_"):
        return _tone_specs(condition.split("_", 1)[1], count, rng)
    if condition == "extended":
        return _extended_specs(count, rng)
    if condition == "wildchat":
        return _wildchat_specs(count, rng, seed=seed)
    raise AssertionError(condition)  # unreachable
