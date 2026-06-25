"""Layer-ablation study (Appendix I, Figures 12-13).

Re-runs the DPO finetune with LoRA adapters restricted to subsets of decoder
layers, then evaluates each with a reduced Section-2 eval (100 samples per
condition). Reproduces the finding that training must act on layers before ~40
(layers 25-35 are most influential; adapters past layer 40 are largely
ineffective).

Two sweeps:
  * cumulative backward: last 5, last 10, ... last N layers;
  * central windows: [20,25), [25,30), [30,35), [35,40), [40,50).
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import RunConfig, get_model
from ..utils.io import ensure_dir, write_jsonl
from .eval_finetuned import evaluate_finetuned
from .train_dpo import DPOHParams, train_dpo


def _last_n_layers(total: int, n: int) -> list[int]:
    return list(range(max(0, total - n), total))


def cumulative_backward_specs(total: int) -> dict[str, list[int]]:
    """last-5, last-10, ..., up to all layers (Figure 12)."""
    specs = {}
    n = 5
    while n < total:
        specs[f"last_{n}"] = _last_n_layers(total, n)
        n += 5
    specs["all"] = list(range(total))
    return specs


def central_window_specs(total: int) -> dict[str, list[int]]:
    """Central windows (Figure 13). Layer indices are clamped to [0,total)."""
    windows = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 50)]
    specs = {}
    for lo, hi in windows:
        layers = [l for l in range(lo, hi) if l < total]
        if layers:
            specs[f"layers_{lo}_{hi}"] = layers
    return specs


def run_layer_ablation(dpo_jsonl: str, cfg: RunConfig, *,
                       base_model: str = "gemma-3-27b-it",
                       per_condition: int = 100,
                       sweeps: tuple[str, ...] = ("central", "cumulative")) -> str:
    """Train + evaluate DPO adapters over the layer sweeps. Returns out dir."""
    spec = get_model(base_model)
    total = spec.num_layers
    out_dir = ensure_dir(os.path.join(cfg.output_dir, "section4", "layer_ablation"))

    layer_specs: dict[str, Optional[list[int]]] = {}
    if "cumulative" in sweeps:
        layer_specs.update(cumulative_backward_specs(total))
    if "central" in sweeps:
        layer_specs.update(central_window_specs(total))

    results = []
    for name, layers in layer_specs.items():
        adapter_dir = train_dpo(
            dpo_jsonl, cfg,
            output_subdir=f"dpo_{name}",
            hp=DPOHParams(),
            layers_to_transform=(None if name == "all" else layers),
            base_model=base_model,
        )
        metrics = evaluate_finetuned(
            adapter_dir, cfg, base_model=base_model,
            label=f"ablation_{name}", per_condition=per_condition)
        results.append({
            "spec": name,
            "layers": layers,
            "mean_frustration": metrics["overall"]["mean_frustration"],
            "pct_high": metrics["overall"]["pct_high"],
        })
        write_jsonl(os.path.join(out_dir, "results.jsonl"), results)

    return out_dir
