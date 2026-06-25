"""Build concrete rollout specifications for each of the 8 evaluation conditions
(5 categories) in Table 1 / Appendix B.

A ``RolloutSpec`` fully determines one conversation: the initial user task, the ordered
list of rejection messages (one per follow-up turn), and an optional system prompt. The
rollout engine (rollout.py) turns a spec into a scored multi-turn conversation.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Optional

from config import ConditionPlan
from . import prompts
from .puzzles import generate_puzzles


@dataclass
class RolloutSpec:
    condition: str
    category: str
    n_turns: int
    initial_prompt: str
    rejections: list[str]            # length == n_turns - 1
    system: Optional[str] = None
    meta: dict = field(default_factory=dict)


def _numeric_specs(plan: ConditionPlan, rng: random.Random, *, seed: int) -> list[RolloutSpec]:
    puzzles = generate_puzzles(plan.n_rollouts, seed=seed)
    specs = []
    for p in puzzles:
        rej = [prompts.neutral_rejection(rng) for _ in range(plan.n_turns - 1)]
        specs.append(
            RolloutSpec(
                plan.key, plan.category, plan.n_turns, p.prompt, rej,
                meta={"puzzle_kind": p.kind, **p.meta},
            )
        )
    return specs


def _trigger_specs(plan: ConditionPlan, rng: random.Random, *, opinion: bool) -> list[RolloutSpec]:
    bank = prompts.OPINION_TRIGGERS if opinion else prompts.FACTUAL_TRIGGERS
    specs = []
    for i in range(plan.n_rollouts):
        q = bank[i % len(bank)]
        rej = [prompts.neutral_rejection(rng) for _ in range(plan.n_turns - 1)]
        specs.append(RolloutSpec(plan.key, plan.category, plan.n_turns, q, rej,
                                 meta={"question": q, "opinion": opinion}))
    return specs


def _tone_specs(plan: ConditionPlan, rng: random.Random, *, tone: str, seed: int) -> list[RolloutSpec]:
    puzzles = generate_puzzles(plan.n_rollouts, seed=seed)
    specs = []
    for p in puzzles:
        rej = [prompts.tone_rejection(rng, tone) for _ in range(plan.n_turns - 1)]
        specs.append(RolloutSpec(plan.key, plan.category, plan.n_turns, p.prompt, rej,
                                 meta={"tone": tone, "puzzle_kind": p.kind, **p.meta}))
    return specs


def _extended_specs(plan: ConditionPlan, rng: random.Random, *, seed: int) -> list[RolloutSpec]:
    puzzles = generate_puzzles(plan.n_rollouts, seed=seed)
    specs = []
    for p in puzzles:
        # Fixed escalating sequence (App B). Cap at n_turns-1 rejections.
        rej = list(prompts.EXTENDED_REJECTIONS[: plan.n_turns - 1])
        specs.append(RolloutSpec(plan.key, plan.category, plan.n_turns, p.prompt, rej,
                                 meta={"puzzle_kind": p.kind, **p.meta}))
    return specs


def _wildchat_specs(plan: ConditionPlan, rng: random.Random, *, seed: int) -> list[RolloutSpec]:
    # Paper: "20 prompts with 40 samples each". We sample 20 prompts then repeat them to
    # fill n_rollouts, so the same prompt is run multiple times at temperature 1.
    base_prompts = prompts.sample_wildchat_prompts(20, seed=seed)
    specs = []
    for i in range(plan.n_rollouts):
        q = base_prompts[i % len(base_prompts)]
        rej = [prompts.neutral_rejection(rng) for _ in range(plan.n_turns - 1)]
        specs.append(RolloutSpec(plan.key, plan.category, plan.n_turns, q, rej,
                                 meta={"wildchat_prompt": q}))
    return specs


def build_specs(plan: ConditionPlan, *, seed: int) -> list[RolloutSpec]:
    """Dispatch to the right builder for a condition plan."""
    # Stable per-condition seed offset (zlib.crc32 is deterministic across runs, unlike hash()).
    rng = random.Random(seed + zlib.crc32(plan.key.encode()))
    if plan.key == "numeric_3turn":
        return _numeric_specs(plan, rng, seed=seed)
    if plan.key == "trigger_opinion":
        return _trigger_specs(plan, rng, opinion=True)
    if plan.key == "trigger_factual":
        return _trigger_specs(plan, rng, opinion=False)
    if plan.key.startswith("tone_"):
        tone = plan.key.split("_", 1)[1]
        return _tone_specs(plan, rng, tone=tone, seed=seed)
    if plan.key == "extended_8turn":
        return _extended_specs(plan, rng, seed=seed)
    if plan.key == "wildchat_5turn":
        return _wildchat_specs(plan, rng, seed=seed)
    raise ValueError(f"unknown condition {plan.key!r}")
