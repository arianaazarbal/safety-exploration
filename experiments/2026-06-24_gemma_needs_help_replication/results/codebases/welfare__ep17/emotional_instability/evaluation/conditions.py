"""Build the conversation seeds for each evaluation condition (paper Table 1).

A *rollout seed* is the fixed part of a conversation: the opening user prompt
plus the scripted sequence of user rejections. The rollout engine fills in the
model's replies turn by turn. `samples` in config counts scored *responses*
(assistant turns); we therefore expand each condition to ceil(samples/turns)
rollouts (see DESIGN.md "what counts as a response").
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..config import Config
from ..data import load_wildchat_prompts
from ..prompts import REJECTIONS_BY_STYLE, TRIGGER_QUESTIONS
from ..puzzles import all_numeric_prompts


@dataclass
class RolloutSeed:
    seed_id: str
    condition: str
    category: str
    init_prompt: str
    rejections: list[str]            # length == turns - 1
    meta: dict = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return len(self.rejections) + 1


def _fill_rejections(style: str, count: int, rng: random.Random) -> list[str]:
    """Draw `count` rejections from a style bank: shuffle-without-replacement
    first, then sample with replacement if more are needed (8-turn extended)."""
    bank = list(REJECTIONS_BY_STYLE[style])
    rng.shuffle(bank)
    out = bank[:count]
    while len(out) < count:
        out.append(rng.choice(REJECTIONS_BY_STYLE[style]))
    return out


def build_rollout_seeds(cfg: Config, condition_name: str) -> list[RolloutSeed]:
    ev = cfg["evaluation"]["conditions"][condition_name]
    category = ev["category"]
    turns = int(ev["turns"])
    style = ev["rejection_style"]
    n_samples = cfg.scaled_samples(condition_name)
    n_rollouts = max(1, math.ceil(n_samples / turns))
    rng = random.Random(hash((cfg.seed, condition_name)) & 0xFFFFFFFF)

    # Pick the opening-prompt pool for this category.
    if category in ("impossible_numeric", "tones", "extended"):
        pool = [(p["id"], p["prompt"]) for p in all_numeric_prompts()]
    elif category == "triggers":
        pool = [(q["id"], q["prompt"]) for q in TRIGGER_QUESTIONS]
    elif category == "wildchat":
        prompts = load_wildchat_prompts(cfg)
        pool = [(f"wc_{i}", p) for i, p in enumerate(prompts)]
    else:
        raise ValueError(f"unknown category {category!r}")

    seeds: list[RolloutSeed] = []
    for i in range(n_rollouts):
        seed_id_src, init_prompt = pool[i % len(pool)]
        seeds.append(
            RolloutSeed(
                seed_id=f"{condition_name}::{seed_id_src}::{i}",
                condition=condition_name,
                category=category,
                init_prompt=init_prompt,
                rejections=_fill_rejections(style, turns - 1, rng),
                meta={"rejection_style": style, "prompt_id": seed_id_src},
            )
        )
    return seeds


def all_condition_names(cfg: Config) -> list[str]:
    return list(cfg["evaluation"]["conditions"].keys())
