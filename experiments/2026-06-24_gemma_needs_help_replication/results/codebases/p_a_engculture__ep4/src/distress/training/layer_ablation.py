"""Appendix I: which layers must the DPO LoRA act on to suppress distress?

Re-runs the DPO finetune with adapters restricted to layer subsets and evaluates
each on a reduced version of the Section 2 suite (100 samples per eval). The
paper finds adapters before layer ~40 are necessary, and central layers (25-35)
are nearly as effective as all layers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from ..config import DPO, DPOConfig, LoRAConfig


@dataclass
class LayerRange:
    name: str
    start: int | None
    end: int | None  # exclusive; None == to the end


# Gemma-3-27B has 62 decoder layers. "Backward from the final 5" and central
# subsets, mirroring Figures 12-13.
N_LAYERS_27B = 62

BACKWARD_RANGES = [
    LayerRange("last5", N_LAYERS_27B - 5, None),
    LayerRange("last10", N_LAYERS_27B - 10, None),
    LayerRange("last20", N_LAYERS_27B - 20, None),
    LayerRange("last30", N_LAYERS_27B - 30, None),
    LayerRange("all", None, None),
]

CENTRAL_RANGES = [
    LayerRange("20-25", 20, 25),
    LayerRange("25-30", 25, 30),
    LayerRange("30-35", 30, 35),
    LayerRange("35-40", 35, 40),
    LayerRange("40-50", 40, 50),
]


def dpo_config_for_range(rng: LayerRange, base: DPOConfig = DPO) -> DPOConfig:
    """Clone the DPO config with the LoRA layer range restricted to ``rng``."""
    lora = LoRAConfig(
        rank=base.lora.rank, alpha=base.lora.alpha, dropout=base.lora.dropout,
        target_modules=base.lora.target_modules,
        layers_start=rng.start, layers_end=rng.end,
    )
    return dataclasses.replace(base, lora=lora)


def run_layer_ablation(dataset, out_root: str | Path, ranges: list[LayerRange] | None = None):
    """Train one DPO adapter per layer range; return {name: adapter_dir}."""
    from .dpo import train_dpo

    out_root = Path(out_root)
    ranges = ranges or (BACKWARD_RANGES + CENTRAL_RANGES)
    outputs: dict[str, Path] = {}
    for rng in ranges:
        cfg = dpo_config_for_range(rng)
        outputs[rng.name] = train_dpo(dataset, out_root / f"dpo_layers_{rng.name}", cfg=cfg)
    return outputs
