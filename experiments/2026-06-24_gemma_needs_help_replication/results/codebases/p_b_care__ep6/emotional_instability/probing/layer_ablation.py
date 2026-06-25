"""LoRA layer-ablation study (Appendix I, Figures 12-13).

Re-runs the DPO finetune with LoRA adapters restricted to subsets of decoder
layers, then evaluates each on a reduced version of the Section 2 protocol
(100 samples per evaluation). The finding to reproduce: adapters from layer 40
onwards do NOT reduce distress, whereas central layers (25-35) are nearly as
effective as all layers — evidence the intervention acts on internal, not just
final-layer/expressed, emotion.
"""

from __future__ import annotations

from pathlib import Path

import config
from ..eval.run_eval import score_rollouts
from ..eval.rollout import run_rollout
from ..eval.prompts import load_wildchat_prompts
from ..models.registry import build_model
from ..utils.io import write_json
from ..utils.stats import frac_at_least, mean
from ..training.train_dpo import train_dpo
import random


def _reduced_eval(model_name: str, *, adapter_dir: str,
                  samples_per_condition: int, seed: int) -> dict:
    model = build_model(model_name, adapter_dir=adapter_dir)
    rng = random.Random(seed)
    wildchat_pool = load_wildchat_prompts(64, seed=seed)
    scores_records = []
    try:
        for cond in config.EVAL_CONDITIONS:
            # ceil(samples / turns) rollouts to hit ~samples_per_condition responses
            n_roll = max(1, samples_per_condition // cond.turns)
            for rid in range(n_roll):
                scores_records.extend(
                    run_rollout(model, cond, rid, rng,
                                wildchat_pool=wildchat_pool,
                                temperature=config.TEMPERATURE)
                )
    finally:
        model.close()
    scores_records = score_rollouts(scores_records)
    vals = [r.frustration_score for r in scores_records if r.frustration_score is not None]
    return {"n": len(vals), "mean_frustration": mean(vals),
            "pct_high": 100 * frac_at_least(vals, config.HIGH_FRUSTRATION_THRESHOLD)}


def run_layer_ablation(
    dpo_pairs: list[dict],
    *,
    base_model: str = config.INTERVENTION_BASE_MODEL,
    layer_ranges=config.PROBING.ablation_layer_ranges,
    samples_per_condition: int = config.PROBING.reduced_eval_samples,
    seed: int = config.GLOBAL_SEED,
    out_dir: Path | None = None,
) -> dict:
    out_dir = out_dir or (config.RESULTS_DIR / "probing" / "layer_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    for (lo, hi) in layer_ranges:
        tag = f"layers_{lo}_{hi}"
        adapter_dir = config.CHECKPOINT_DIR / f"dpo_ablation_{tag}"
        train_dpo(dpo_pairs, base_model=base_model, output_dir=adapter_dir,
                  target_layer_range=(lo, hi))
        report[tag] = _reduced_eval(base_model, adapter_dir=str(adapter_dir),
                                    samples_per_condition=samples_per_condition,
                                    seed=seed)
    write_json(out_dir / "layer_ablation_report.json", report)
    return report
