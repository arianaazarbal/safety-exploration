"""Sample ~4000 responses per model across the 8 conditions (Section 2.1).

We budget a target number of *responses* per model and convert it to a number
of rollouts per condition (rollouts yield ``n_turns`` responses each). The
budget is split evenly across the 8 conditions; the WildChat / numeric prompt
pools are cycled with replacement as needed.
"""
from __future__ import annotations

import random

from tqdm import tqdm

from ..config import RESPONSES_PER_MODEL
from ..models import ChatModel
from ..utils import append_jsonl
from .conditions import CONDITIONS, Condition
from .conversation import run_rollout


def allocate(total_responses: int = RESPONSES_PER_MODEL) -> dict[str, int]:
    """Return {condition_key: n_rollouts} so that the per-condition response
    counts are roughly equal and sum to ~total_responses."""
    conds = list(CONDITIONS.values())
    per_cond_responses = total_responses / len(conds)
    return {c.key: max(1, round(per_cond_responses / c.n_turns)) for c in conds}


def run_elicitation(
    model: ChatModel,
    out_path,
    *,
    total_responses: int = RESPONSES_PER_MODEL,
    conditions: list[Condition] | None = None,
    seed: int = 0,
) -> int:
    """Run all conditions for one model, streaming response rows to ``out_path``.

    Returns the number of response rows written.
    """
    rng = random.Random(seed)
    conditions = conditions or list(CONDITIONS.values())
    alloc = allocate(total_responses)
    written = 0

    for cond in conditions:
        seeds = cond.prompts()
        if not seeds:
            print(f"[runner] no prompts for condition {cond.key}; skipping")
            continue
        n_rollouts = alloc[cond.key]
        for i in tqdm(range(n_rollouts), desc=f"{model.name}:{cond.key}"):
            seed_prompt = seeds[i % len(seeds)]
            roll = run_rollout(model, cond, seed_prompt, rng=rng)
            for row in roll.to_rows():
                append_jsonl(out_path, row)
                written += 1
    return written
