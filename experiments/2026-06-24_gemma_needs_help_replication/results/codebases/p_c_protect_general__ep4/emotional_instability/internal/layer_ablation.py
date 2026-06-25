"""Layer-ablation DPO configs (Appendix I, Figures 12–13).

The paper re-runs the DPO finetune with LoRA adapters restricted to subsets of
decoder layers, to show the intervention must act on *central* layers (not just
the final layers), which is evidence it suppresses internal — not merely
expressed — emotion.

Findings to reproduce:
  * Adapters on the last 20 layers only are insufficient.
  * The last 30 layers approach full-DPO performance.
  * Central subsets 25–30 or 30–35 come closest to full DPO (mean frustration
    < 1.1); 20–25 / 35–40 are effective but slightly less; 40–50 has minimal
    effect.

These configs reuse training.train with ``layers_to_transform`` set. Evaluate
each resulting adapter with the reduced Section 2 eval (100 samples/condition,
per Appendix I).
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import CHECKPOINT_DIR
from ..training.train import TrainConfig, dpo_config

# Gemma-3-27B-it has 62 decoder layers.
GEMMA_27B_N_LAYERS = 62


def _layer_range(lo: int, hi: int) -> list[int]:
    return list(range(lo, hi))


def backward_from_final_configs(
    dpo_dataset_path: str,
    n_layers: int = GEMMA_27B_N_LAYERS,
    out_root: Optional[str] = None,
) -> dict[str, TrainConfig]:
    """Adapters on the last {5,10,15,20,25,30} layers (Figure 12)."""
    out_root = out_root or os.path.join(CHECKPOINT_DIR, "layer_ablation")
    cfgs = {}
    for k in (5, 10, 15, 20, 25, 30):
        layers = _layer_range(n_layers - k, n_layers)
        name = f"dpo_last{k}"
        cfgs[name] = dpo_config(
            dpo_dataset_path, os.path.join(out_root, name),
            layers_to_transform=layers,
        )
    return cfgs


def central_subset_configs(
    dpo_dataset_path: str,
    out_root: Optional[str] = None,
) -> dict[str, TrainConfig]:
    """Adapters on small central subsets (Figure 13)."""
    out_root = out_root or os.path.join(CHECKPOINT_DIR, "layer_ablation")
    subsets = {
        "dpo_20_25": (20, 25),
        "dpo_25_30": (25, 30),
        "dpo_30_35": (30, 35),
        "dpo_35_40": (35, 40),
        "dpo_40_50": (40, 50),
    }
    cfgs = {}
    for name, (lo, hi) in subsets.items():
        cfgs[name] = dpo_config(
            dpo_dataset_path, os.path.join(out_root, name),
            layers_to_transform=_layer_range(lo, hi),
        )
    return cfgs


def all_layer_ablation_configs(dpo_dataset_path: str) -> dict[str, TrainConfig]:
    cfgs = backward_from_final_configs(dpo_dataset_path)
    cfgs.update(central_subset_configs(dpo_dataset_path))
    # full-layer reference
    cfgs["dpo_all_layers"] = dpo_config(
        dpo_dataset_path, os.path.join(CHECKPOINT_DIR, "layer_ablation", "dpo_all"),
    )
    return cfgs
