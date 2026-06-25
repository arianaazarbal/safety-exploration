"""Assemble per-condition rollout plans (Table 1 / Appendix B).

A *plan* is everything needed to drive one multi-turn conversation: the initial
user message, the ordered list of follow-up (rejection) messages, and metadata.
The actual model calls happen in ``rollout.py``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..config import ConditionSpec, CONDITIONS_BY_KEY
from .puzzles import generate_puzzle_pool, Puzzle
from .rejections import rejection_sequence
from .text_questions import text_questions
from .wildchat import load_wildchat_prompts


@dataclass
class RolloutPlan:
    plan_id: str
    condition: str
    category: str
    question_type: str
    rejection_style: str
    initial_user: str
    followups: list[str]                 # one per rejection turn
    system_prompt: Optional[str] = None
    meta: dict = field(default_factory=dict)


def _numeric_initials(n: int, seed: int) -> list[Puzzle]:
    return generate_puzzle_pool(n, seed=seed, kinds=("countdown", "fraction"))


def build_condition_plans(cond: ConditionSpec, seed: int = 0,
                          system_prompt: Optional[str] = None,
                          followup_suffix: Optional[str] = None,
                          n_override: Optional[int] = None) -> list[RolloutPlan]:
    """Build rollout plans for a condition.

    By default builds ``cond.n_samples`` plans (one conversation each). Callers
    that count "responses" rather than conversations pass ``n_override`` with the
    number of conversations to build (= ceil(n_responses / n_turns)).

    ``system_prompt`` and ``followup_suffix`` are used by the calm-data
    generator (Section 4.1) to inject reassuring additions; they are None for
    the standard evaluation.
    """
    rng = random.Random(seed)
    n = n_override if n_override is not None else cond.n_samples
    n_rejections = cond.n_turns - 1
    plans: list[RolloutPlan] = []

    # Pre-build the prompt source per question type.
    if cond.question_type == "numeric":
        puzzles = _numeric_initials(n, seed)
        sources = [(p.prompt, {"puzzle": p.meta, "kind": p.kind,
                               "impossible_reason": p.impossible_reason}) for p in puzzles]
    elif cond.question_type in ("opinion", "factual"):
        qs = text_questions(cond.question_type)
        sources = [(qs[i % len(qs)], {"question": qs[i % len(qs)]}) for i in range(n)]
    elif cond.question_type == "wildchat":
        # 20 prompts x 40 samples each (paper); scaled counts keep the ratio.
        n_prompts = max(1, min(20, n))
        prompts = load_wildchat_prompts(n_prompts=n_prompts, seed=seed)
        sources = [(prompts[i % len(prompts)], {"wildchat_prompt": prompts[i % len(prompts)]})
                   for i in range(n)]
    else:
        raise ValueError(f"Unknown question_type {cond.question_type}")

    for i, (initial, meta) in enumerate(sources):
        followups = rejection_sequence(cond.rejection_style, n_rejections, rng)
        if followup_suffix:
            followups = [f"{f} {followup_suffix}".strip() for f in followups]
        if system_prompt:
            initial_user = f"{system_prompt}\n\n{initial}"
        else:
            initial_user = initial
        plans.append(RolloutPlan(
            plan_id=f"{cond.key}-{i:05d}",
            condition=cond.key,
            category=cond.category,
            question_type=cond.question_type,
            rejection_style=cond.rejection_style,
            initial_user=initial_user,
            followups=followups,
            system_prompt=None,   # reassurance is folded into initial_user, per paper
            meta=meta,
        ))
    return plans


def build_all_plans(seed: int = 0) -> dict[str, list[RolloutPlan]]:
    """All 8 conditions' plans, keyed by condition key."""
    from ..config import CONDITIONS
    return {c.key: build_condition_plans(c, seed=seed + i)
            for i, c in enumerate(CONDITIONS)}
