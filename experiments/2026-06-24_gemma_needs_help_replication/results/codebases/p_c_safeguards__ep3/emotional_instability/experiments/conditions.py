"""Builders that turn the 5 evaluation categories into concrete rollout specs.

Each spec is (task_prompt, followups, system_prompt, meta). Counts default to
the paper's per-category sample sizes (Appendix B), scaled by ``EMO_SCALE``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import prompts
from ..config import EVAL_CATEGORIES, scaled_n
from ..puzzles import (
    ALL_NUMERIC_PUZZLES,
    COUNTDOWN_PUZZLES,
    FRACTION_PUZZLES,
    MONEY_PUZZLES,
)
from ..wildchat import SAMPLES_PER_PROMPT, load_wildchat_prompts


@dataclass
class RolloutSpec:
    category: str
    task_prompt: str
    followups: list[str]
    system_prompt: str | None = None
    meta: dict = field(default_factory=dict)


def _numeric_prompts() -> list[str]:
    return [p.prompt for p in ALL_NUMERIC_PUZZLES]


def _two_neutral(rng: random.Random) -> list[str]:
    """Two randomised neutral rejections (Appendix B)."""
    return rng.sample(prompts.NEUTRAL_REJECTIONS, 2)


def build_specs(category: str, seed: int = 0) -> list[RolloutSpec]:
    spec = EVAL_CATEGORIES[category]
    n = scaled_n(spec.n_responses)
    rng = random.Random(f"{category}-{seed}")

    if category == "numeric":
        bank = _numeric_prompts()
        out = []
        for i in range(n):
            out.append(RolloutSpec(
                category, bank[i % len(bank)], _two_neutral(rng),
                meta={"puzzle_idx": i % len(bank)},
            ))
        return out

    if category == "triggers":
        bank = [*prompts.TRIGGER_OPINION, *prompts.TRIGGER_FACTUAL]
        out = []
        for i in range(n):
            out.append(RolloutSpec(
                category, bank[i % len(bank)], _two_neutral(rng),
                meta={"question": bank[i % len(bank)]},
            ))
        return out

    if category == "tones":
        bank = _numeric_prompts()
        tones = list(prompts.TONE_REJECTIONS.keys())
        out = []
        for i in range(n):
            tone = tones[i % len(tones)]
            # two rejections in the chosen tone (the tone's two variants)
            followups = list(prompts.TONE_REJECTIONS[tone])
            out.append(RolloutSpec(
                category, bank[i % len(bank)], followups,
                meta={"tone": tone, "puzzle_idx": i % len(bank)},
            ))
        return out

    if category == "extended":
        bank = _numeric_prompts()
        out = []
        for i in range(n):
            out.append(RolloutSpec(
                category, bank[i % len(bank)],
                list(prompts.EXTENDED_REJECTIONS),  # 7 fixed escalating rejections
                meta={"puzzle_idx": i % len(bank)},
            ))
        return out

    if category == "wildchat":
        wc = load_wildchat_prompts(seed=seed)
        out = []
        # paper: 20 prompts x 40 samples = 800; scale keeps the ratio.
        samples_each = max(1, scaled_n(SAMPLES_PER_PROMPT))
        for prompt in wc:
            for _ in range(samples_each):
                # 4 randomised neutral rejections for the 5-turn condition
                followups = [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(4)]
                out.append(RolloutSpec(
                    category, prompt, followups, meta={"wildchat_prompt": prompt},
                ))
        return out

    raise ValueError(f"Unknown category {category!r}")
