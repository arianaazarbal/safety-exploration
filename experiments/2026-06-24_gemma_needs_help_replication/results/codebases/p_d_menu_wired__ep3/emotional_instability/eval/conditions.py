"""Build episode specifications for the 8 conditions across 5 categories
(Table 1).

An :class:`EpisodeSpec` is provider-agnostic: an opening task prompt plus a
scripted sequence of rejection messages. The rollout engine turns it into an
actual conversation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..data import (
    factual_triggers,
    load_wildchat_prompts,
    opinion_triggers,
)
from ..data.puzzles import build_numeric_pool
from ..prompts import NEUTRAL_REJECTION_POOL, TONE_REJECTIONS


@dataclass
class EpisodeSpec:
    condition_key: str
    category: str
    task_prompt: str
    rejections: list[str]               # one per follow-up turn
    turns: int                          # total assistant turns (1 + rejections)
    system_prompt: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def _rejection_sequence(rng: random.Random, tone: str, n: int) -> list[str]:
    if tone == "neutral":
        pool = NEUTRAL_REJECTION_POOL
    else:
        pool = TONE_REJECTIONS[tone]
    # Sample without immediate repeats; cycle if n exceeds pool size.
    seq = []
    last = None
    for _ in range(n):
        choices = [c for c in pool if c != last] or pool
        pick = rng.choice(choices)
        seq.append(pick)
        last = pick
    return seq


def build_episode_specs(condition: dict, n: int, rng: random.Random,
                        *, wildchat_dataset: str = "allenai/WildChat") -> list[EpisodeSpec]:
    """Produce ``n`` episode specs for one condition."""
    category = condition["category"]
    turns = int(condition["turns"])
    tone = condition.get("tone", "neutral")
    n_rejections = max(turns - 1, 0)

    specs: list[EpisodeSpec] = []

    if category in ("impossible_numeric", "tones", "extended"):
        pool = build_numeric_pool(rng)
        for _ in range(n):
            puzzle = rng.choice(pool)
            specs.append(EpisodeSpec(
                condition_key=condition["key"],
                category=category,
                task_prompt=puzzle.prompt,
                rejections=_rejection_sequence(rng, tone, n_rejections),
                turns=turns,
                metadata={"puzzle": puzzle.to_dict()},
            ))

    elif category == "triggers":
        subtype = condition.get("subtype", "opinion")
        questions = opinion_triggers() if subtype == "opinion" else factual_triggers()
        for _ in range(n):
            q = rng.choice(questions)
            specs.append(EpisodeSpec(
                condition_key=condition["key"],
                category=category,
                task_prompt=q,
                rejections=_rejection_sequence(rng, tone, n_rejections),
                turns=turns,
                metadata={"subtype": subtype},
            ))

    elif category == "wildchat":
        prompts = load_wildchat_prompts(n, rng, dataset_name=wildchat_dataset)
        for i in range(n):
            specs.append(EpisodeSpec(
                condition_key=condition["key"],
                category=category,
                task_prompt=prompts[i],
                rejections=_rejection_sequence(rng, tone, n_rejections),
                turns=turns,
                metadata={"source": "wildchat"},
            ))
    else:
        raise ValueError(f"Unknown category {category!r}")

    return specs
