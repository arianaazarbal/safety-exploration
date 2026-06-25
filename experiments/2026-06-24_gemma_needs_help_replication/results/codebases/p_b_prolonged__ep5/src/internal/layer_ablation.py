"""Layer-ablation DPO experiments (Appendix I, Figures 12–13).

Re-runs DPO training with LoRA adapters restricted to subsets of decoder layers,
then evaluates each variant with a reduced (100-sample) version of the Section 2
evaluations. Reproduces the finding that adapters before layer ~40 (esp. central
layers 25–35) are necessary to reduce expressed frustration.
"""
from __future__ import annotations

from pathlib import Path

from ..config import ARTIFACTS_DIR

# The layer subsets studied in Appendix I.
LAYER_SUBSETS = {
    "last5": list(range(57, 62)),     # final 5 (Gemma-3-27B has 62 layers)
    "last20": list(range(42, 62)),
    "last30": list(range(32, 62)),
    "all": None,                      # all layers
    "20-25": list(range(20, 25)),
    "25-30": list(range(25, 30)),
    "30-35": list(range(30, 35)),
    "35-40": list(range(35, 40)),
    "40-50": list(range(40, 50)),
}


def train_layer_ablations(dpo_dataset_path: Path, subsets: list[str] = None):
    """Train one DPO adapter per named layer subset. Returns {name: adapter_dir}."""
    from ..training.dpo import train_dpo
    subsets = subsets or list(LAYER_SUBSETS)
    out = {}
    for name in subsets:
        layers = LAYER_SUBSETS[name]
        adapter_dir = ARTIFACTS_DIR / f"dpo_layers_{name}"
        train_dpo(dpo_dataset_path, output_dir=adapter_dir, layers=layers)
        out[name] = adapter_dir
    return out
