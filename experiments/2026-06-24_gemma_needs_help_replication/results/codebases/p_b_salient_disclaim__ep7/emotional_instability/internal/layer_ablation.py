"""Layer-ablation DPO sweep (Appendix I, Figures 12-13).

Re-run the DPO finetune with LoRA adapters restricted to subsets of decoder
layers, then evaluate each resulting adapter with the reduced Section 2 protocol
(100 samples per evaluation) to see which layers must be intervened on to reduce
expressed frustration.

Two sweeps (config.INTERNAL):
  * backward_sweeps: last-N layers (working backward from the final layer).
  * central_subsets: small central windows, e.g. (25,30), (30,35), (40,50).

This module only orchestrates training+eval of the adapters; the actual training
uses training.dpo_train.train_dpo(target_layers=...).
"""

from __future__ import annotations

import os
from typing import Optional

import config
from ..training.dpo_train import train_dpo
from ..training.calm_data import PreferencePair


def _num_layers(base_model: str) -> int:
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(base_model).num_hidden_layers


def backward_layer_windows(base_model: str) -> list[tuple[int, int]]:
    """Return [lo, hi) windows for the last-N-layers backward sweep."""
    n = _num_layers(base_model)
    return [(max(0, n - k), n) for k in config.INTERNAL.backward_sweeps]


def central_layer_windows() -> list[tuple[int, int]]:
    return [tuple(w) for w in config.INTERNAL.central_subsets]


def run_layer_ablation(pairs: list[PreferencePair], *,
                       base_model: str = "google/gemma-3-27b-it",
                       output_root: Optional[str] = None,
                       include_backward: bool = True,
                       include_central: bool = True,
                       seed: int = config.SEED) -> dict:
    """Train a DPO adapter for each layer window and return the adapter paths.

    Evaluation of each adapter is done by the run_internal.py script (it reuses
    the Section 2 eval with ABLATION_N_SAMPLES); here we just produce the
    adapters keyed by their layer window.
    """
    output_root = output_root or os.path.join(config.OUTPUT_DIR, "layer-ablation")
    windows: list[tuple[int, int]] = []
    if include_backward:
        windows += backward_layer_windows(base_model)
    if include_central:
        windows += central_layer_windows()

    adapters: dict[str, str] = {}
    for (lo, hi) in windows:
        out_dir = os.path.join(output_root, f"dpo-layers-{lo}-{hi}")
        train_dpo(pairs, base_model=base_model, output_dir=out_dir,
                  target_layers=(lo, hi), seed=seed)
        adapters[f"{lo}-{hi}"] = out_dir
    return adapters
