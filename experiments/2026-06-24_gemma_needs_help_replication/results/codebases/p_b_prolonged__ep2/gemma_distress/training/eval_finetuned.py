"""Evaluate a finetuned (LoRA-adapter) Gemma model with the Section-2 protocol.

Reused by the DPO/SFT comparison (Figure 5) and the layer ablation (Appendix I,
which uses a reduced eval of 100 samples per condition). Builds an HFBackend
with the adapter loaded and runs the standard rollout + judge pipeline.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Optional

from ..config import RunConfig, get_model
from ..eval.analysis import headline_metrics
from ..eval.conditions import build_conditions
from ..eval.judge_runner import FrustrationJudge
from ..eval.run_eval import generate_rollouts
from ..models.hf_backend import HFBackend
from ..utils.io import ensure_dir, write_jsonl


def reduced_sample_counts(per_condition: int) -> dict:
    """A reduced budget with ~`per_condition` rollouts per category-condition.

    The layer ablation uses 100 samples per evaluation (Appendix I). Because we
    split categories into 1-3 conditions, we set each category budget so each
    condition lands near `per_condition` rollouts.
    """
    return {
        "impossible_numeric": per_condition,
        "triggers": per_condition * 2,     # 2 conditions
        "tones": per_condition * 3,        # 3 conditions
        "extended": per_condition,
        "wildchat": per_condition,
    }


def evaluate_finetuned(adapter_path: Optional[str], cfg: RunConfig, *,
                       base_model: str = "gemma-3-27b-it",
                       label: str = "finetuned",
                       per_condition: Optional[int] = None,
                       judge: bool = True) -> dict:
    """Run Section-2 eval on a (possibly adapter-loaded) Gemma model.

    `adapter_path=None` evaluates the vanilla base instruct model (useful as the
    Figure-5 reference). `per_condition` overrides the sample budget for a quick
    reduced eval (e.g. 100 for layer ablation).

    Returns the headline-metrics dict and writes rollouts + summary to disk.
    """
    spec = get_model(base_model)
    backend = HFBackend(spec, cfg, adapter_path=adapter_path)

    counts = (reduced_sample_counts(per_condition)
              if per_condition is not None else dict(cfg.sample_counts))
    local_cfg = dataclasses.replace(cfg, sample_counts=counts)
    conditions = build_conditions(counts)

    records = generate_rollouts(base_model, local_cfg, conditions=conditions,
                                backend=backend)
    if judge:
        FrustrationJudge(cfg).score_rollouts(records)

    out_dir = ensure_dir(os.path.join(cfg.output_dir, "section4", "eval", label))
    write_jsonl(os.path.join(out_dir, "rollouts.jsonl"),
                [r.to_row() for r in records])
    metrics = headline_metrics([r.to_row() for r in records])
    import json
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
