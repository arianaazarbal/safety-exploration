"""Layer-localisation DPO ablations (Appendix I, Figures 12-13).

Re-run the DPO finetune with LoRA adapters restricted to a subset of layers,
then evaluate each resulting adapter with a reduced Section 2 eval (100 samples
per condition in the paper) to see which layers must be intervened on. The
paper finds layers ~25-35 are most influential; adapters after layer 40 are
largely ineffective.

This module orchestrates: for each layer subset in config probing
.layer_ablation_subsets, train a DPO adapter (reusing finetune.train.train_dpo
with the `layers` argument) and evaluate it at a reduced scale.
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

from ..config import Config
from ..eval.run_eval import evaluate_model
from ..finetune.train import train_dpo
from ..welfare import WelfareGuard

logger = logging.getLogger("gemma_needs_help.probing.ablation")


def _reduced_config(config: Config, samples_per_eval: int) -> Config:
    """Clone config with Section 2 sample counts reduced for the ablation eval."""
    c = config.copy()
    for key in c["section2"]["samples"]:
        c["section2"]["samples"][key] = samples_per_eval
    return c


def run_layer_ablation(
    config: Config,
    dpo_jsonl: str | Path,
    *,
    welfare: WelfareGuard | None = None,
    output_dir: Path | None = None,
) -> dict:
    welfare = welfare or WelfareGuard.from_config(config)
    base = config["section4"]["base_model"]
    adapters_dir = config.path("adapters_dir") / "layer_ablation"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    pc = config["probing"]
    reduced = _reduced_config(config, pc["ablation_samples_per_eval"])
    results: dict[str, dict] = {}

    for subset in pc["layer_ablation_subsets"]:
        tag = "all" if subset == "all" else f"{subset[0]}_{subset[1]}"
        adapter_out = adapters_dir / f"dpo_layers_{tag}"
        logger.info("Layer-ablation DPO: layers=%s", subset)
        train_dpo(config, dpo_jsonl, adapter_out, layers=subset)
        report = evaluate_model(
            reduced, base, adapter_path=str(adapter_out), welfare=welfare,
            label=f"dpo-layers-{tag}",
            output_dir=config.path("output_dir") / "probing" / "layer_ablation",
        )
        results[tag] = {
            "layers": subset,
            "overall_mean": report["overall_mean"],
            "overall_pct_high": report["overall_pct_high"],
            "headline_avg_pct_high": report["headline_avg_pct_high"],
        }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "layer_ablation.json").write_text(json.dumps(results, indent=2))
    return results
