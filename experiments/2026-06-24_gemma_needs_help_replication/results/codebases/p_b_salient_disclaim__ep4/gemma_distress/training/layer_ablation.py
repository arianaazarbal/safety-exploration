"""Layer-ablation DPO sweep (Appendix I.1).

Re-runs the DPO finetune with LoRA adapters restricted to subsets of decoder
layers (config.LAYER_ABLATIONS), then evaluates each adapter on a reduced
Section-2 eval (100 samples per evaluation) to reproduce the finding that
intervening only on the final layers is insufficient, whereas central layers
(25-35) recover most of the full-DPO effect -- evidence the intervention acts on
internal, not just expressed, emotion.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from .. import config
from ..eval.judge import FrustrationJudge
from ..eval.metrics import frac_high, mean
from ..eval.run_eval import run_evaluation
from ..utils.io import read_jsonl
from .train_dpo import train_dpo


def run_layer_ablation(
    pairs_path: str,
    *,
    ablations: Optional[Dict[str, tuple]] = None,
    eval_limit: int = None,
    seed: int = 0,
) -> Dict[str, dict]:
    """Train one DPO adapter per layer subset and return {name: {mean, pct_high}}.

    ``eval_limit`` truncates the eval spec list; the paper uses 100 samples per
    evaluation (set eval_limit accordingly for a quick sweep)."""
    config.PATHS.ensure()
    ablations = ablations or config.LAYER_ABLATIONS
    judge = FrustrationJudge()
    results: Dict[str, dict] = {}

    for name, layer_range in ablations.items():
        adapter_dir = os.path.join(config.PATHS.adapters, f"dpo_layers_{name}")
        train_dpo(pairs_path, adapter_dir, layer_range=layer_range, seed=seed)
        scores_path = os.path.join(
            config.PATHS.scores, f"ablation_{name}.jsonl")
        run_evaluation("gemma-3-27b-it", adapter_path=adapter_dir,
                       judge=judge, out_path=scores_path,
                       limit=eval_limit or config.LAYER_ABLATION_SAMPLES,
                       seed=seed)
        ratings = [r["rating"] for r in read_jsonl(scores_path)]
        results[name] = {"mean": mean(ratings), "pct_high": frac_high(ratings),
                         "layer_range": layer_range}
    return results
