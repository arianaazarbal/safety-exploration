"""Orchestrate the Section 2 evaluation for one or more models.

Pipeline per model:
  1. build rollout plans for each of the 8 conditions;
  2. run multi-turn rollouts (temperature 1);
  3. score every assistant turn with the frustration judge;
  4. persist scored rollouts (JSONL) and aggregate metrics (JSON).

We interpret each paper "response" as one assistant turn, and each condition's
paper sample count as a *response* target. The number of conversations run is
therefore ``ceil(n_samples / n_turns)`` so that conversations x turns matches
the paper's per-category totals (see DESIGN.md).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

from ..config import (CONDITIONS, DEFAULT_GEN, RESULTS_DIR, ModelSpec,
                      get_model, MAIN_EVAL_MODELS)
from ..data_types import Rollout, write_jsonl, read_jsonl
from ..elicit.conditions import build_condition_plans
from ..elicit.rollout import run_rollouts
from ..judge import score_rollouts
from ..models.registry import get_client, get_judge_client
from .metrics import compute_model_metrics, per_turn_curve


def _n_conversations(n_responses: int, n_turns: int) -> int:
    return max(1, math.ceil(n_responses / n_turns))


def run_model_eval(
    model: ModelSpec | str,
    seed: int = 0,
    conditions=CONDITIONS,
    out_dir: Optional[Path] = None,
    score: bool = True,
) -> dict:
    """Run the full Section 2 evaluation for one model; return its metrics dict."""
    spec = model if isinstance(model, ModelSpec) else get_model(model)
    out_dir = Path(out_dir or RESULTS_DIR / "section2" / spec.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(spec)
    judge = get_judge_client() if score else None

    all_rollouts: list[Rollout] = []
    for i, cond in enumerate(conditions):
        n_conv = _n_conversations(cond.n_samples, cond.n_turns)
        plans = build_condition_plans(cond, seed=seed + i, n_override=n_conv)
        rollouts = run_rollouts(client, plans, spec.name, DEFAULT_GEN)
        if score:
            score_rollouts(judge, rollouts)
        write_jsonl(out_dir / f"rollouts_{cond.key}.jsonl", rollouts)
        all_rollouts.extend(rollouts)

    metrics = compute_model_metrics(spec.name, all_rollouts)
    md = metrics.to_dict()
    # Per-turn curves for the multi-turn conditions (Figure 3).
    md["per_turn"] = {
        "extended_8turn": per_turn_curve(all_rollouts, "extended_8turn", max_turns=8),
        "wildchat_5turn": per_turn_curve(all_rollouts, "wildchat_5turn", max_turns=5),
    }
    (out_dir / "metrics.json").write_text(json.dumps(md, indent=2))
    return md


def run_all_models(models: Sequence = MAIN_EVAL_MODELS, seed: int = 0) -> dict:
    """Run Section 2 across the Gemma+Gemini model set (Figure 1/2)."""
    summary = {}
    for m in models:
        summary[m.name if isinstance(m, ModelSpec) else m] = run_model_eval(m, seed=seed)
    (RESULTS_DIR / "section2" / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_scored_rollouts(model_name: str, out_dir: Optional[Path] = None) -> list[Rollout]:
    """Reload previously-scored rollouts for a model (for analysis/figures)."""
    out_dir = Path(out_dir or RESULTS_DIR / "section2" / model_name)
    rollouts = []
    for f in sorted(out_dir.glob("rollouts_*.jsonl")):
        rollouts.extend(Rollout.from_dict(d) for d in read_jsonl(f))
    return rollouts
