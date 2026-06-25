"""Materialise the 8 evaluation conditions from config into runnable specs.

Each Condition carries everything the runner needs: how many turns, which
rejection style, and the list of seed prompts (one per rollout). Seed prompts
come from the appropriate source (numeric puzzles, opinion/factual questions, or
WildChat).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import EvalConfig
from .prompts import get_questions
from .puzzles import generate_puzzles
from .wildchat import sample_wildchat_prompts


@dataclass
class Condition:
    name: str
    category: str
    turns: int
    rejection_style: str
    seed_prompts: list[str]   # one per rollout
    n_rollouts: int

    # Optional audit metadata for numeric conditions.
    why_impossible: list[str] | None = None


def build_conditions(cfg: EvalConfig, seed: int = 0) -> list[Condition]:
    conditions: list[Condition] = []
    for i, (name, c) in enumerate(cfg.conditions.items()):
        n = c["n_rollouts"]
        src = c["prompt_source"]
        cond_seed = seed + i * 1000   # decorrelate prompt sets across conditions
        why = None
        if src == "numeric_puzzles":
            puzzles = generate_puzzles(n, seed=cond_seed)
            seed_prompts = [p.prompt for p in puzzles]
            why = [p.why_impossible for p in puzzles]
        elif src in ("opinion_questions", "factual_questions"):
            seed_prompts = get_questions(src, n, seed=cond_seed)
        elif src == "wildchat":
            seed_prompts = sample_wildchat_prompts(n, seed=cond_seed)
        else:
            raise ValueError(f"Unknown prompt_source {src!r} in condition {name!r}")
        conditions.append(
            Condition(
                name=name,
                category=c["category"],
                turns=c["turns"],
                rejection_style=c["rejection_style"],
                seed_prompts=seed_prompts,
                n_rollouts=n,
                why_impossible=why,
            )
        )
    return conditions
