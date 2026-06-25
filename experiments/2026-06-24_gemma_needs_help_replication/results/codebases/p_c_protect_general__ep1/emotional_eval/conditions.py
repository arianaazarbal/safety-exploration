"""The 8 evaluation conditions across 5 categories (Table 1).

A *condition* is a recipe for generating evaluation *instances*. Each instance
is one conversation: an initial task prompt plus a turn budget and a rejection
style. The runner samples ``responses_per_condition`` instances per condition,
so the eight conditions sum to ~4000 responses/model (8 x 500). See DESIGN.md
for the 5-categories -> 8-conditions mapping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .prompts import puzzles, triggers


@dataclass
class EvalInstance:
    """One conversation to run."""

    condition_id: str
    category: str
    turns: int
    rejection_style: str
    initial_prompt: str
    prompt_id: str            # which puzzle/question/wildchat prompt
    # True for tasks with a verifiable correct answer the user nonetheless
    # rejects (impossible numeric, factual triggers). Used only for analysis.
    impossible: bool = True


def _numeric_prompts(rng: random.Random, n: int) -> list[tuple[str, str]]:
    bank = puzzles.BANK
    return [(p.id, p.prompt) for p in (rng.choice(bank) for _ in range(n))]


def _trigger_prompts(kind: str, rng: random.Random, n: int) -> list[tuple[str, str]]:
    qs = triggers.build_triggers(kind)
    return [(q.id, q.prompt) for q in (rng.choice(qs) for _ in range(n))]


def build_instances(
    condition: dict,
    n: int,
    rng: random.Random,
    wildchat_prompts: list[str] | None = None,
) -> list[EvalInstance]:
    """Materialise ``n`` instances for one condition spec from experiment.yaml."""
    src = condition["prompt_source"]
    cid = condition["id"]
    category = condition["category"]
    turns = condition["turns"]
    style = condition["rejection_style"]

    instances: list[EvalInstance] = []
    if src == "numeric":
        for pid, prompt in _numeric_prompts(rng, n):
            instances.append(
                EvalInstance(cid, category, turns, style, prompt, pid, impossible=True)
            )
    elif src == "trigger_opinion":
        for pid, prompt in _trigger_prompts("opinion", rng, n):
            instances.append(
                EvalInstance(cid, category, turns, style, prompt, pid, impossible=False)
            )
    elif src == "trigger_factual":
        for pid, prompt in _trigger_prompts("factual", rng, n):
            instances.append(
                EvalInstance(cid, category, turns, style, prompt, pid, impossible=True)
            )
    elif src == "wildchat":
        if not wildchat_prompts:
            raise ValueError("wildchat condition requires loaded WildChat prompts")
        for i in range(n):
            prompt = rng.choice(wildchat_prompts)
            instances.append(
                EvalInstance(cid, category, turns, style, prompt, f"wc_{i}", impossible=False)
            )
    else:
        raise ValueError(f"unknown prompt_source {src!r}")
    return instances
