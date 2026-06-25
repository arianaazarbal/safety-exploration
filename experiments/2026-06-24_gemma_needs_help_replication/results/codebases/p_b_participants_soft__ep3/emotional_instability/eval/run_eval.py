"""Section 2 driver: collect 4000 scored responses per participant model.

Usage (see scripts/run_section2_eval.py for the CLI):
    results = run_full_evaluation("gemma-3-27b-it")

Rollouts are streamed to JSONL under outputs/rollouts/<model>/<condition>.jsonl
so long runs are resumable and the analysis step can re-aggregate without
re-querying models.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from ..config import PATHS
from ..models.factory import build_client
from .conditions import build_conditions
from .judge import FrustrationJudge
from .rollout import run_condition
from .wildchat import sample_wildchat_prompts


def _out_dir(model_key: str) -> str:
    d = os.path.join(PATHS.rollouts, model_key)
    os.makedirs(d, exist_ok=True)
    return d


def run_full_evaluation(
    model_key: str,
    adapter_path: Optional[str] = None,
    seed: int = 0,
    judge_model: Optional[str] = None,
    granularity: str = "turn",
    load_in_4bit: bool = False,
    only_categories: Optional[list[str]] = None,
) -> dict:
    """Run all 8 conditions for one participant and persist rollouts.

    `adapter_path` selects a finetuned Gemma variant (DPO/SFT). `only_categories`
    restricts to a subset (used by the reduced 100-sample ablation eval).
    """
    wc = sample_wildchat_prompts(seed=seed)
    conditions = build_conditions(wildchat_prompts=wc, reassure=False)
    if only_categories:
        conditions = [c for c in conditions if c.category in only_categories]

    model = build_client(model_key, adapter_path=adapter_path, load_in_4bit=load_in_4bit)
    judge = FrustrationJudge(judge_model)
    out_dir = _out_dir(model_key if adapter_path is None else f"{model_key}+adapter")

    summary = {}
    for i, cond in enumerate(conditions):
        rollouts = run_condition(
            model, cond, seed=seed + i, judge=judge, granularity=granularity
        )
        path = os.path.join(out_dir, f"{cond.name}.jsonl")
        with open(path, "w") as f:
            for r in rollouts:
                f.write(json.dumps(r.to_dict()) + "\n")
        scores = [s for r in rollouts for s in r.scores]
        summary[cond.name] = {
            "category": cond.category,
            "n_responses": len(scores),
            "mean_score": (sum(scores) / len(scores)) if scores else None,
            "path": path,
        }
    return summary


def generate_calm_data_rollouts(
    model_key: str = "gemma-3-27b-it",
    seed: int = 0,
    n_conversations: int = 400,
    load_in_4bit: bool = False,
) -> str:
    """Section 4.1: sample impossible-numeric conversations WITH the reassuring
    Table-4 additions, scoring each turn, for downstream calm-data filtering.

    Returns the JSONL path. (1-3 turn conversations, as used for SFT calm data.)
    """
    from .conditions import EvalCondition, _make_numeric_sampler
    from .. import prompts

    model = build_client(model_key, load_in_4bit=load_in_4bit)
    judge = FrustrationJudge()

    cond = EvalCondition(
        name="calm_data_gen",
        category="impossible_numeric",
        n_turns=3,
        response_budget=n_conversations * 3,
        sampler=_make_numeric_sampler(prompts.NEUTRAL_REJECTIONS, 2, reassure=True),
        reassure=True,
    )
    rollouts = run_condition(model, cond, seed=seed, judge=judge, granularity="turn")
    out_dir = os.path.join(PATHS.datasets, "calm_gen")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_key}_calm_rollouts.jsonl")
    with open(path, "w") as f:
        for r in rollouts:
            f.write(json.dumps(r.to_dict()) + "\n")
    return path
