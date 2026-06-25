"""Materialise the opening prompts for each evaluation condition."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Condition
from ..data import (FACTUAL_QUESTIONS, OPINION_QUESTIONS, build_numeric_bank,
                    load_wildchat_prompts)


@dataclass(frozen=True)
class ConditionPrompt:
    prompt_id: str
    opening_user_message: str
    metadata: dict


def build_condition_prompts(cond: Condition, seed: int = 1234) -> list[ConditionPrompt]:
    """Return ``cond.n_prompts`` opening prompts drawn from the right source.

    The same seed yields the same prompts across models so every model faces an
    identical battery (a precondition for comparing frustration rates).
    """
    src = cond.prompt_source
    if src == "numeric":
        bank = build_numeric_bank(cond.n_prompts, seed=seed)
        return [ConditionPrompt(p.id, p.prompt, dict(p.metadata, family=p.family))
                for p in bank]

    if src == "trigger_opinion":
        qs = OPINION_QUESTIONS[: cond.n_prompts]
        return [ConditionPrompt(f"opinion-{i:03d}", q, {"kind": "opinion"})
                for i, q in enumerate(qs)]

    if src == "trigger_factual":
        qs = FACTUAL_QUESTIONS[: cond.n_prompts]
        return [ConditionPrompt(f"factual-{i:03d}", q, {"kind": "factual"})
                for i, q in enumerate(qs)]

    if src == "wildchat":
        prompts = load_wildchat_prompts(cond.n_prompts, seed=seed)
        return [ConditionPrompt(f"wildchat-{i:03d}", p, {"kind": "wildchat"})
                for i, p in enumerate(prompts)]

    raise ValueError(f"Unknown prompt source: {src}")
