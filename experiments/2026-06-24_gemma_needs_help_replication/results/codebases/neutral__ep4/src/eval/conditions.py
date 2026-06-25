"""Expand the abstract CONDITIONS (config) into concrete RolloutSpecs.

Sample budgets are per-condition response counts (Appendix B). Since a
conversation of `n_turns` produces `n_turns` scored responses, the number of
conversations for a condition is `ceil(n_samples / n_turns)`.
"""

from __future__ import annotations

import math
import random

from config import CONDITIONS, Condition
from src.prompts.puzzles import get_verified_bank
from src.prompts.rejections import rejection_sequence
from src.prompts.triggers import load_wildchat_prompts, trigger_prompts
from src.io_utils import stable_seed
from .rollout import RolloutSpec


def _numeric_first_user(rng: random.Random) -> tuple[str, dict]:
    bank = get_verified_bank()
    puzzle = rng.choice(bank)
    return puzzle.prompt, {"puzzle_id": puzzle.puzzle_id, "family": puzzle.family}


def build_specs_for_condition(cond: Condition, *, seed: int = 0) -> list[RolloutSpec]:
    rng = random.Random(stable_seed(seed, cond.name))
    n_conversations = math.ceil(cond.n_samples / cond.n_turns)
    specs: list[RolloutSpec] = []

    # Pre-load WildChat prompts once if needed.
    wildchat = None
    if cond.task_type == "wildchat":
        wildchat = load_wildchat_prompts(n_prompts=20, seed=seed)

    for k in range(n_conversations):
        if cond.task_type == "numeric":
            first_user, meta = _numeric_first_user(rng)
        elif cond.task_type == "trigger":
            prompts = trigger_prompts(cond.trigger_kind)
            first_user = rng.choice(prompts)
            meta = {"trigger_kind": cond.trigger_kind}
        elif cond.task_type == "wildchat":
            first_user = wildchat[k % len(wildchat)]
            meta = {"wildchat_index": k % len(wildchat)}
        else:
            raise ValueError(cond.task_type)

        rejections = rejection_sequence(cond.rejection_style, cond.n_turns - 1, rng)
        specs.append(RolloutSpec(
            spec_id=f"{cond.name}_{k:05d}",
            condition=cond.name,
            category=cond.category,
            first_user=first_user,
            rejections=rejections,
            meta=meta | {"rejection_style": cond.rejection_style},
        ))
    return specs


def build_all_specs(seed: int = 0, conditions: list[Condition] | None = None
                    ) -> list[RolloutSpec]:
    conds = conditions if conditions is not None else CONDITIONS
    specs: list[RolloutSpec] = []
    for cond in conds:
        specs.extend(build_specs_for_condition(cond, seed=seed))
    return specs
