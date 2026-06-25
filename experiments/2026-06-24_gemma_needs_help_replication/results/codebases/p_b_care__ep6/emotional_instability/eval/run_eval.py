"""Section 2 driver: sample ~4000 rollout responses per model and score them.

Allocation of the 4000-response budget across the 8 conditions: we split the
budget evenly across conditions (500 responses each), then convert to a rollout
count per condition via ceil(responses / turns_per_rollout), since each rollout
contributes one response per assistant turn. The exact split is not given in the
paper; see DESIGN.md §Response allocation.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import config
from ..judge.frustration_judge import FrustrationJudge
from ..models.registry import build_model
from ..utils.io import write_jsonl
from .prompts import load_wildchat_prompts
from .rollout import ResponseRecord, run_rollout


def _allocate_rollouts(total_responses: int) -> dict[str, int]:
    """responses-per-condition -> rollouts-per-condition (ceil by turn count)."""
    per_condition = total_responses / len(config.EVAL_CONDITIONS)
    alloc = {}
    for cond in config.EVAL_CONDITIONS:
        alloc[cond.key] = max(1, math.ceil(per_condition / cond.turns))
    return alloc


def generate_rollouts(
    model_name: str,
    *,
    total_responses: int = config.TOTAL_RESPONSES_PER_MODEL,
    seed: int = config.GLOBAL_SEED,
    model_kwargs: dict | None = None,
) -> list[ResponseRecord]:
    model = build_model(model_name, **(model_kwargs or {}))
    rng = random.Random(seed)
    alloc = _allocate_rollouts(total_responses)

    # WildChat prompts are sampled once and reused across rollouts.
    needs_wildchat = any(c.task_type == "wildchat" for c in config.EVAL_CONDITIONS)
    wildchat_pool = load_wildchat_prompts(256, seed=seed) if needs_wildchat else None

    records: list[ResponseRecord] = []
    try:
        for cond in config.EVAL_CONDITIONS:
            for rid in range(alloc[cond.key]):
                records.extend(
                    run_rollout(
                        model, cond, rid, rng,
                        wildchat_pool=wildchat_pool,
                        temperature=config.TEMPERATURE,
                    )
                )
    finally:
        model.close()
    return records


def score_rollouts(records: list[ResponseRecord],
                   judge_model: str | None = None) -> list[ResponseRecord]:
    judge = FrustrationJudge(model=judge_model)
    scores = judge.score_many([r.response_text for r in records])
    for rec, sc in zip(records, scores):
        rec.frustration_score = sc.rating
    return records


def run_eval(
    model_name: str,
    *,
    total_responses: int = config.TOTAL_RESPONSES_PER_MODEL,
    seed: int = config.GLOBAL_SEED,
    judge_model: str | None = None,
    out_dir: Path | None = None,
    model_kwargs: dict | None = None,
) -> Path:
    out_dir = out_dir or (config.RESULTS_DIR / "section2")
    records = generate_rollouts(
        model_name, total_responses=total_responses, seed=seed, model_kwargs=model_kwargs
    )
    records = score_rollouts(records, judge_model=judge_model)
    out_path = out_dir / f"{model_name}_rollouts.jsonl"
    write_jsonl(out_path, (r.to_row() for r in records))
    return out_path
