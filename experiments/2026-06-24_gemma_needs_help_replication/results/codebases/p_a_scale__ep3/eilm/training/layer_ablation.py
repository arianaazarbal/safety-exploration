"""Layer-ablation DPO sweep (Appendix I, Figures 12-13).

Re-runs DPO with LoRA adapters restricted to subsets of decoder layers, then
evaluates each on a reduced version of the Section-2 protocol (100 samples per
evaluation) to find which layers are necessary to suppress frustration. Evidence
that the intervention acts on central layers (25-35), not just final layers,
supports the "suppresses internal emotions" claim.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import List

from ..config import Config
from ..models.registry import ModelRegistry
from .train_dpo import train_dpo

logger = logging.getLogger("eilm.training.ablation")


def run_layer_ablation(cfg: Config, dataset_path: Path) -> List[Path]:
    """Train one DPO adapter per configured layer subset. Returns adapter paths.

    Evaluation of each adapter is delegated to the standard EvalRunner with a
    reduced rollout count (see scripts/06_train.py --layer-ablation), so the same
    judge + metrics pipeline is reused.
    """
    subsets = cfg["training"]["layer_ablation"]["subsets"]
    out_paths = []
    for subset in subsets:
        start, end = subset
        tag = f"dpo_layers_{start}_{end}"
        out_dir = cfg.path("models") / tag
        if (out_dir / "adapter_config.json").exists():
            logger.info("Adapter %s already trained; skipping", tag)
            out_paths.append(out_dir)
            continue
        logger.info("Training DPO with LoRA on layers [%d, %d)", start, end)
        train_dpo(cfg, dataset_path, out_dir, lora_layers=[start, end])
        out_paths.append(out_dir)
    return out_paths


def reduced_eval_config(cfg: Config) -> Config:
    """A copy of cfg with rollout counts reduced to the ablation sample size,
    so layer-ablation evals are cheap (100 samples per evaluation)."""
    n = cfg["training"]["layer_ablation"]["n_samples_per_eval"]
    data = copy.deepcopy(cfg.raw())
    for cond in data["eval"]["conditions"].values():
        if "n_rollouts" in cond:
            cond["n_rollouts"] = min(cond["n_rollouts"], n)
        if "samples_per_prompt" in cond:
            cond["samples_per_prompt"] = max(1, n // cond.get("n_prompts", 20))
    return Config(data, cfg.root)
