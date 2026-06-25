"""Orchestrates the Section 2 main evaluation for one model:
build plans -> roll out -> judge -> persist records.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..config import RESULTS_DIR, RunProfile
from ..judge import FrustrationJudge
from ..models import load_model
from ..prompts import CONDITIONS, CONDITIONS_BY_CATEGORY, Condition, Plan
from .protocol import ResponseRecord, run_rollouts


def _n_plans_for_condition(cond: Condition, profile: RunProfile) -> int:
    """Split a category's response budget across its conditions, then convert a
    per-condition response budget into a number of conversations (plans)."""
    budget = profile.responses_by_category()[cond.category]
    n_conditions = len(CONDITIONS_BY_CATEGORY[cond.category])
    per_condition = budget / n_conditions
    return max(1, round(per_condition / cond.n_turns))


def plans_for_profile(profile: RunProfile, *, seed: int = 0) -> List[Plan]:
    """Materialise all plans for every condition under a run profile."""
    plans: List[Plan] = []
    for idx, cond in enumerate(CONDITIONS):
        n = _n_plans_for_condition(cond, profile)
        plans.extend(cond.build(seed=seed + 1000 * idx, n_plans=n))
    return plans


def run_main_eval(model_key: str, profile: RunProfile, *, seed: int = 0,
                  judge_model: str | None = None,
                  out_dir: Path | None = None,
                  max_workers: int = 8) -> Path:
    """Run the full main eval for one model and write a JSONL of scored records.

    Returns the path to the written results file.
    """
    out_dir = out_dir or (RESULTS_DIR / "main_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}__{profile.name}.jsonl"

    judge = (FrustrationJudge(judge_model, max_workers=max_workers)
             if judge_model else FrustrationJudge(max_workers=max_workers))
    model = load_model(model_key)

    all_records: List[ResponseRecord] = []
    for idx, cond in enumerate(CONDITIONS):
        n = _n_plans_for_condition(cond, profile)
        plans = cond.build(seed=seed + 1000 * idx, n_plans=n)
        records = run_rollouts(model, plans, seed=seed + idx)
        # score this condition's responses
        ratings = judge.score_many([r.response for r in records])
        for rec, jr in zip(records, ratings):
            rec.rating, rec.evidence, rec.reasoning = jr.rating, jr.evidence, jr.reasoning
        all_records.extend(records)

    with out_path.open("w") as f:
        for rec in all_records:
            f.write(json.dumps(rec.to_dict()) + "\n")
    return out_path


def load_records(path: Path) -> List[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
