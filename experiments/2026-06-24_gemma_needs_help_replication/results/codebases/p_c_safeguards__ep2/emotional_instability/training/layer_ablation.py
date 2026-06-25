"""Layer-subset DPO ablation (Appendix I, Figures 12-13).

Re-runs the DPO finetune with LoRA adapters restricted to a subset of decoder
layers, to test *where* the intervention must act.  The paper's finding: layers
prior to ~40 are necessary; adapters on layers 30-35 alone are nearly as
effective as all layers, while adapters on 40+ are largely ineffective — taken
as evidence the intervention acts on internal (central-layer) emotional states
rather than only on final-layer expression.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..config import Config
from .build_dataset import DPOExample
from .dpo import train_dpo

# Layer windows probed in Appendix I (inclusive ranges -> explicit layer lists).
ABLATION_WINDOWS: dict[str, range] = {
    "all": range(0, 0),          # sentinel: handled as None (all layers)
    "last5": range(57, 62),      # final 5 (gemma-3-27b has 62 layers)
    "last20": range(42, 62),
    "last30": range(32, 62),
    "L20-25": range(20, 25),
    "L25-30": range(25, 30),
    "L30-35": range(30, 35),
    "L35-40": range(35, 40),
    "L40-50": range(40, 50),
}


def config_for_window(config: Config, window: str) -> Config:
    """Return a config whose LoRA adapters are restricted to ``window``."""
    rng = ABLATION_WINDOWS[window]
    layers = None if window == "all" else tuple(rng)
    return replace(config, lora=replace(config.lora, layers=layers))


def train_dpo_window(
    examples: list[DPOExample],
    config: Config,
    window: str,
    base_model_id: str = "google/gemma-3-27b-it",
    output_dir: str | Path | None = None,
) -> str:
    """Train one layer-ablation DPO variant; returns the adapter path."""
    ablated = config_for_window(config, window)
    out = output_dir or (config.paths.checkpoints / f"dpo_{window}")
    return train_dpo(examples, ablated, base_model_id=base_model_id, output_dir=out)
