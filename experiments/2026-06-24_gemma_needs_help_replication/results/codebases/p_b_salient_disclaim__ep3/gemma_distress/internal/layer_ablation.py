"""Layer-subset DPO ablation (Appendix I, Figures 12-13).

Train the DPO intervention with LoRA adapters restricted to a subset of decoder
layers, then evaluate with the reduced Section-2 protocol (100 samples per
eval). The paper finds layers 25-35 are most influential and that adapters after
layer 40 are largely ineffective — evidence the intervention acts on internal
states, not just final-layer expression.
"""

from __future__ import annotations

from pathlib import Path

import config
from ..training.lora import lora_with_layers
from ..training.train_dpo import train_dpo


def train_layer_variant(dpo_pairs: list[dict], layer_range: tuple[int, int],
                        *, base_lora: config.LoRAConfig | None = None) -> Path:
    base_lora = base_lora or config.DPO.lora
    lo, hi = layer_range
    out = config.ADAPTERS_DIR / f"dpo-layers-{lo}-{hi}"
    return train_dpo(
        dpo_pairs,
        out_dir=out,
        lora_override=lora_with_layers(base_lora, layer_range),
    )


def run_layer_ablation(
    dpo_pairs: list[dict],
    *,
    grids: tuple[tuple[int, int], ...] | None = None,
) -> list[dict]:
    """Train a DPO adapter for each layer range. Returns adapter paths to be
    evaluated (reduced protocol) by the caller / script."""
    grids = grids or (config.INTERNAL.backward_layer_grid + config.INTERNAL.central_layer_grid)
    results = []
    for layer_range in grids:
        adapter = train_layer_variant(dpo_pairs, layer_range)
        results.append({"layer_range": layer_range, "adapter_path": str(adapter)})
    return results
