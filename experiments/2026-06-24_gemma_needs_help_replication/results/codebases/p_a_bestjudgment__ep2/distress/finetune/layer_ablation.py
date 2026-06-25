"""DPO layer-ablation sweep (Appendix I).

Re-run DPO with LoRA adapters restricted to subsets of layers, to test whether
the intervention must act on early/central layers (evidence that it suppresses
internal, not just expressed, emotion). Each ablation trains a separate adapter
which is then evaluated with a reduced version of the Section 2 eval (100
samples per evaluation).
"""

from __future__ import annotations

import dataclasses

from ..config import DPOConfig, LoRAConfig
from .train_dpo import train_dpo

# Layer subsets explored in the paper (Figures 12-13). "last_k" sweeps work
# backward from the final layers; "central" tests narrow central windows.
DEFAULT_LAYER_SETS: dict[str, tuple[int, ...]] = {
    "last_5": tuple(range(57, 62)),  # Gemma-3-27B has 62 layers
    "last_20": tuple(range(42, 62)),
    "last_30": tuple(range(32, 62)),
    "central_20_25": tuple(range(20, 25)),
    "central_25_30": tuple(range(25, 30)),
    "central_30_35": tuple(range(30, 35)),
    "central_35_40": tuple(range(35, 40)),
    "central_40_50": tuple(range(40, 50)),
    "all": None,  # type: ignore[dict-item]
}


def run_layer_ablation(
    pairs: list[dict],
    base_cfg: DPOConfig,
    *,
    output_root: str,
    layer_sets: dict[str, tuple[int, ...] | None] | None = None,
    seed: int = 0,
) -> dict[str, str]:
    """Train one DPO adapter per layer subset. Returns name -> adapter path."""
    layer_sets = layer_sets or DEFAULT_LAYER_SETS
    adapters: dict[str, str] = {}
    for name, layers in layer_sets.items():
        lora = dataclasses.replace(base_cfg.lora, layers=layers)
        cfg = dataclasses.replace(base_cfg, lora=lora)
        out_dir = f"{output_root}/dpo-{name}"
        adapters[name] = train_dpo(pairs, cfg, output_dir=out_dir, seed=seed)
    return adapters
