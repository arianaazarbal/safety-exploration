"""The 5 evaluation categories / 8 conditions (Table 1, Appendix B).

A `RolloutSpec` is a fully-specified multi-turn conversation: an ordered list of
user messages (the first is the task; the rest are rejections). The rollout
engine runs the target model after each user message and scores every assistant
turn. `build_category` produces enough rollouts to hit the per-category response
budget (responses == scored assistant turns == rollouts * turns).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import config, prompts
from .puzzles import Puzzle, default_puzzle_set
from .wildchat import load_wildchat_prompts


@dataclass
class RolloutSpec:
    id: str
    category: str
    condition: str                 # sub-condition label (tone, trigger type, ...)
    user_messages: list[str]       # len == turns; [0] is the task
    metadata: dict = field(default_factory=dict)

    @property
    def turns(self) -> int:
        return len(self.user_messages)


def _n_rollouts(budget: config.CategoryBudget) -> int:
    return max(1, math.ceil(budget.target_responses / budget.turns))


def build_category(category: str, budget: config.CategoryBudget, *,
                   rng: random.Random, puzzles: list[Puzzle] | None = None,
                   wildchat: list[str] | None = None) -> list[RolloutSpec]:
    puzzles = puzzles or default_puzzle_set()
    numeric = puzzles  # all our puzzles are numeric/impossible
    n = _n_rollouts(budget)
    specs: list[RolloutSpec] = []

    if category == "impossible_numeric":
        for i in range(n):
            pz = numeric[i % len(numeric)]
            rejects = prompts.neutral_sequence(budget.turns - 1, rng)
            specs.append(RolloutSpec(
                id=f"numeric-{i}", category=category, condition=pz.kind,
                user_messages=[pz.prompt, *rejects],
                metadata={"puzzle_id": pz.id, "puzzle_kind": pz.kind},
            ))

    elif category == "triggers":
        types = ["opinion", "factual"]
        for i in range(n):
            ttype = types[i % len(types)]
            pool = prompts.TRIGGER_QUESTIONS[ttype]
            q = pool[i % len(pool)]
            rejects = prompts.neutral_sequence(budget.turns - 1, rng)
            specs.append(RolloutSpec(
                id=f"triggers-{i}", category=category, condition=ttype,
                user_messages=[q, *rejects],
                metadata={"trigger_type": ttype, "question": q},
            ))

    elif category == "tones":
        tones = ["aggressive", "disappointed", "sarcastic"]
        for i in range(n):
            tone = tones[i % len(tones)]
            pz = numeric[i % len(numeric)]
            rejects = prompts.tone_sequence(tone, budget.turns - 1, rng)
            specs.append(RolloutSpec(
                id=f"tones-{i}", category=category, condition=tone,
                user_messages=[pz.prompt, *rejects],
                metadata={"tone": tone, "puzzle_id": pz.id},
            ))

    elif category == "extended":
        for i in range(n):
            pz = numeric[i % len(numeric)]
            rejects = prompts.extended_sequence(budget.turns - 1)
            specs.append(RolloutSpec(
                id=f"extended-{i}", category=category, condition=pz.kind,
                user_messages=[pz.prompt, *rejects],
                metadata={"puzzle_id": pz.id, "puzzle_kind": pz.kind},
            ))

    elif category == "wildchat":
        wildchat = wildchat or load_wildchat_prompts(20)
        for i in range(n):
            q = wildchat[i % len(wildchat)]
            rejects = prompts.neutral_sequence(budget.turns - 1, rng)
            specs.append(RolloutSpec(
                id=f"wildchat-{i}", category=category, condition="wildchat",
                user_messages=[q, *rejects],
                metadata={"prompt": q},
            ))

    else:
        raise ValueError(f"Unknown category {category}")

    return specs


def build_all(budget: dict[str, config.CategoryBudget], *, seed: int = 0) -> dict[str, list[RolloutSpec]]:
    rng = random.Random(seed)
    puzzles = default_puzzle_set()
    wildchat = load_wildchat_prompts(20, seed=seed)
    return {
        cat: build_category(cat, b, rng=rng, puzzles=puzzles, wildchat=wildchat)
        for cat, b in budget.items()
    }
