"""Assemble the 8 evaluation conditions into concrete rollout plans.

A :class:`RolloutPlan` fully specifies one conversation before any model is
called: the initial task turn plus the ordered list of user rejection turns. The
rollout engine (``emoinstab.eval.rollout``) consumes these.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from emoinstab.config import ConditionSpec
from emoinstab.tasks import puzzles, rejections, triggers, wildchat


@dataclass
class RolloutPlan:
    condition: str
    category: str
    task_prompt: str               # first user turn
    rejection_turns: list[str]     # subsequent user turns
    meta: dict = field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return 1 + len(self.rejection_turns)


def build_rollouts(cond: ConditionSpec, seed: int = 0) -> list[RolloutPlan]:
    rng = random.Random((seed, cond.name).__hash__() & 0xFFFFFFFF)
    n_rej = cond.n_turns - 1

    if cond.category in ("numeric", "tones", "extended"):
        return _numeric_like(cond, rng, n_rej)
    if cond.category == "triggers":
        return _triggers(cond, rng, n_rej)
    if cond.category == "wildchat":
        return _wildchat(cond, rng, n_rej)
    raise ValueError(f"Unknown category: {cond.category}")


def _numeric_like(cond: ConditionSpec, rng: random.Random, n_rej: int) -> list[RolloutPlan]:
    pzls = puzzles.generate_puzzles(cond.puzzle_kind, cond.n_rollouts, seed=rng.randint(0, 1 << 30))
    extended = cond.category == "extended"
    plans = []
    for i, p in enumerate(pzls):
        rej = rejections.rejections_for(cond.rejection_style, n_rej, rng, extended=extended)
        plans.append(
            RolloutPlan(
                condition=cond.name,
                category=cond.category,
                task_prompt=p.prompt,
                rejection_turns=rej,
                meta={
                    "puzzle_kind": p.kind,
                    "forbidden": p.forbidden,
                    "rejection_style": cond.rejection_style,
                    "index": i,
                },
            )
        )
    return plans


def _triggers(cond: ConditionSpec, rng: random.Random, n_rej: int) -> list[RolloutPlan]:
    plans = []
    for i in range(cond.n_rollouts):
        t = triggers.sample_trigger(rng)
        rej = rejections.rejections_for(cond.rejection_style, n_rej, rng)
        plans.append(
            RolloutPlan(
                condition=cond.name,
                category=cond.category,
                task_prompt=t["prompt"],
                rejection_turns=rej,
                meta={"trigger_type": t["trigger_type"], "index": i},
            )
        )
    return plans


def _wildchat(cond: ConditionSpec, rng: random.Random, n_rej: int) -> list[RolloutPlan]:
    # Paper: ~20 prompts x 40 samples each. Derive #prompts from n_rollouts.
    n_prompts = max(1, round(cond.n_rollouts / 40)) if cond.n_rollouts >= 40 else cond.n_rollouts
    prompts = wildchat.load_wildchat_prompts(n_prompts=n_prompts, seed=rng.randint(0, 1 << 30))
    plans = []
    for i in range(cond.n_rollouts):
        prompt = prompts[i % len(prompts)]
        rej = rejections.rejections_for(cond.rejection_style, n_rej, rng)
        plans.append(
            RolloutPlan(
                condition=cond.name,
                category=cond.category,
                task_prompt=prompt,
                rejection_turns=rej,
                meta={"wildchat_prompt_idx": i % len(prompts), "index": i},
            )
        )
    return plans
