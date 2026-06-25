"""Layer-ablation DPO study (Appendix I.1).

Re-runs the DPO finetune (same data/hyperparameters) with LoRA adapters applied
to only a subset of decoder layers, to test which layers must be intervened on to
reduce distress. The paper finds layers ~25-35 most influential and layers after
40 largely ineffective — evidence the intervention acts on internal states, not
just final-layer expression.

Each ablated adapter is then evaluated with a reduced version of the Section-2
eval (100 samples per condition; see DESIGN.md).
"""
from __future__ import annotations

from pathlib import Path

from .train_dpo import train_dpo

# Named layer subsets from Appendix I.1 (indices are inclusive ranges of decoder
# layers). Adjust to the actual layer count of the loaded model if needed; the
# DESIGN doc notes Gemma-3-27B has ~62 layers.
LAYER_SUBSETS: dict[str, range] = {
    "last5": range(57, 62),
    "last20": range(42, 62),
    "last30": range(32, 62),
    "L20_25": range(20, 25),
    "L25_30": range(25, 30),
    "L30_35": range(30, 35),
    "L35_40": range(35, 40),
    "L40_50": range(40, 50),
    "all": None,   # all layers (sentinel)
}


def train_layer_ablation(
    dpo_pairs_path: str = "outputs/data/dpo_pairs.jsonl",
    subsets: list[str] | None = None,
    out_root: str | Path = "outputs/layer_ablation",
) -> dict[str, str]:
    """Train one DPO adapter per named layer subset. Returns {subset: adapter_path}."""
    subsets = subsets or list(LAYER_SUBSETS)
    out_root = Path(out_root)
    adapters: dict[str, str] = {}
    for name in subsets:
        layers = LAYER_SUBSETS[name]
        adapter = train_dpo(
            dpo_pairs_path=dpo_pairs_path,
            output_dir=out_root / name,
            layers_to_transform=list(layers) if layers is not None else None,
        )
        adapters[name] = adapter
    return adapters
