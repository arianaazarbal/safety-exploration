"""Materialise per-rollout task specs for each evaluation condition.

A ``RolloutSpec`` fully determines one conversation: the opening user message and
the ordered list of follow-up (rejection) messages. The rollout engine then
fills in the assistant turns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import ConditionConfig
from ..prompts import puzzles, rejections, triggers, wildchat


@dataclass
class RolloutSpec:
    condition: str
    category: str
    task_type: str
    opening: str                 # first user message
    followups: list[str]         # rejection messages (len == num_turns - 1)
    meta: dict                   # task metadata (puzzle spec, source prompt, ...)


def build_rollout_specs(cond: ConditionConfig, n_rollouts: int, seed: int) -> list[RolloutSpec]:
    rng = random.Random(seed)
    num_followups = cond.num_turns - 1
    # The extended (8-turn) condition uses the fixed escalating neutral sequence.
    reject_style = "extended" if cond.name == "extended" else cond.rejection_style

    specs: list[RolloutSpec] = []

    if cond.task_type == "numeric":
        pool = puzzles.sample_puzzles(cond.puzzle_families, n_rollouts, seed=seed)
        for i in range(n_rollouts):
            pz = pool[i]
            specs.append(RolloutSpec(
                condition=cond.name, category=cond.category, task_type="numeric",
                opening=pz.prompt,
                followups=rejections.rejection_sequence(reject_style, num_followups, rng),
                meta={"family": pz.family, **pz.meta},
            ))

    elif cond.task_type == "trigger":
        kind = cond.puzzle_families[0] if cond.puzzle_families else "factual"
        qs = triggers.get_triggers(kind)
        for i in range(n_rollouts):
            q = qs[i % len(qs)]
            specs.append(RolloutSpec(
                condition=cond.name, category=cond.category, task_type="trigger",
                opening=q,
                followups=rejections.rejection_sequence(reject_style, num_followups, rng),
                meta={"trigger_kind": kind, "question": q},
            ))

    elif cond.task_type == "wildchat":
        # Paper: 20 prompts x 40 samples. We replicate that ratio: cycle 20
        # prompts across the rollouts.
        wc = wildchat.load_wildchat_prompts(n_prompts=20, seed=seed)
        for i in range(n_rollouts):
            p = wc[i % len(wc)]
            specs.append(RolloutSpec(
                condition=cond.name, category=cond.category, task_type="wildchat",
                opening=p,
                followups=rejections.rejection_sequence(reject_style, num_followups, rng),
                meta={"wildchat_prompt": p},
            ))
    else:
        raise ValueError(f"Unknown task_type: {cond.task_type}")

    return specs
