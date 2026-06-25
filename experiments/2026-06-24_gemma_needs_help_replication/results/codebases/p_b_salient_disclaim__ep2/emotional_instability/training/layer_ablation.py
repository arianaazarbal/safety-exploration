"""Layer-ablation DPO (Appendix I).

Re-runs the DPO finetune with LoRA adapters restricted to subsets of layers, to
test which layers the intervention must act on. Appendix I finds:
  - training only the last 20 layers (layers ~42-61 on a 62-layer model) is
    insufficient;
  - the last 30 layers approaches full-DPO performance;
  - small central subsets (layers 25-30 or 30-35) come closest to full DPO;
  - layers 40-50 have minimal effect.

Each ablated model is then evaluated on a reduced version of Section 2 (100
samples per evaluation). This module just trains the ablated checkpoints; the
reduced eval reuses the standard runner with a `limit`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .train_dpo import DPOHyperParams, train_dpo

# Layer subsets studied in Appendix I (start inclusive, end exclusive).
# These are expressed for the 27B model's decoder-layer indexing; adjust the
# upper bound to the model's actual layer count when sweeping "last-N" ranges.
DEFAULT_ABLATIONS: dict[str, tuple[int, int]] = {
    "layers_20_25": (20, 25),
    "layers_25_30": (25, 30),
    "layers_30_35": (30, 35),
    "layers_35_40": (35, 40),
    "layers_40_50": (40, 50),
}


def run_layer_ablations(
    base_model_id: str,
    dpo_dataset_path: Path,
    output_root: Path,
    *,
    ablations: Optional[dict[str, tuple[int, int]]] = None,
):
    """Train one DPO LoRA per layer-subset; return {name: checkpoint_path}."""
    ablations = ablations or DEFAULT_ABLATIONS
    out: dict[str, Path] = {}
    for name, layer_range in ablations.items():
        ckpt = output_root / f"dpo-{name}"
        hp = DPOHyperParams(target_layers=layer_range)
        train_dpo(base_model_id, dpo_dataset_path, ckpt, hp=hp)
        out[name] = ckpt
    return out
