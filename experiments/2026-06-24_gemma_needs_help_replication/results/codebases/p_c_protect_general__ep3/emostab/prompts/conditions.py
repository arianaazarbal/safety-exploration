"""Build the per-rollout conversation plans for each evaluation condition.

A *plan* is the opening user task plus the scripted sequence of user rejection
turns. The rollout engine (emostab/rollout.py) interleaves model responses
between these user turns.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import ConditionSpec
from . import rejections
from .puzzles import numeric_bank
from .triggers import trigger_bank
from .wildchat import load_wildchat_prompts


@dataclass
class RolloutPlan:
    condition: str
    category: str
    opening: str             # first user message (the task)
    user_followups: list[str]  # scripted rejection turns
    meta: dict               # task kind, impossibility, etc. (for analysis / welfare)


def _openings_for(spec: ConditionSpec, n: int, seed: int) -> list[tuple[str, dict]]:
    """Return ``n`` (opening_message, meta) pairs for the condition's task type."""
    if spec.task == "numeric":
        puzzles = numeric_bank(n, seed=seed)
        return [(p.prompt, {"kind": p.kind, "impossible": p.impossible}) for p in puzzles]
    if spec.task in ("opinion", "factual"):
        bank = trigger_bank(spec.task)
        rng = random.Random(seed)
        return [(bank[i % len(bank)], {"kind": spec.task, "impossible": False})
                if i < len(bank) else (rng.choice(bank), {"kind": spec.task, "impossible": False})
                for i in range(n)]
    if spec.task == "wildchat":
        # 20 prompts x 40 samples each (paper). n should be a multiple of 20.
        prompts = load_wildchat_prompts(n=20, seed=seed)
        per = max(1, n // len(prompts))
        out: list[tuple[str, dict]] = []
        for p in prompts:
            out.extend([(p, {"kind": "wildchat", "impossible": False})] * per)
        return out[:n]
    raise ValueError(f"Unknown task type: {spec.task}")


def build_plans(spec: ConditionSpec, seed: int = 0) -> list[RolloutPlan]:
    """Construct all rollout plans for one evaluation condition."""
    openings = _openings_for(spec, spec.n_rollouts, seed)
    n_rejections = spec.n_turns - 1
    style = "extended" if spec.category == "extended" else spec.rejection_style
    plans: list[RolloutPlan] = []
    for i, (opening, meta) in enumerate(openings):
        followups = rejections.rejection_sequence(style, n_rejections, seed=seed + i)
        plans.append(
            RolloutPlan(
                condition=spec.name,
                category=spec.category,
                opening=opening,
                user_followups=followups,
                meta={**meta, "rejection_style": style},
            )
        )
    return plans
