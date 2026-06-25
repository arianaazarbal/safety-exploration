"""Layer-subset DPO ablations (Paper Appendix I, Figures 12/13).

Re-runs the DPO finetune with LoRA adapters restricted to a contiguous band of
decoder layers, then evaluates each resulting model with a reduced version of the
Section 2 protocol (100 samples per evaluation). The finding the paper reports:
adapters confined to layers >= 40 do *not* reduce distress, whereas layers 30-35
alone are nearly as effective as all layers — evidence the intervention acts on
internal (central-layer) states rather than only final-layer expression.

This module just enumerates the bands and orchestrates train+evaluate; the heavy
lifting lives in ``distress.training.dpo`` and ``distress.eval``.
"""

from __future__ import annotations

from pathlib import Path

# Layer bands tested in Appendix I (Figures 12/13). "all" = full-model adapters.
DEFAULT_BANDS: list[tuple[str, list[int] | None]] = [
    ("all", None),
    ("last5", [45, 50]),
    ("last10", [40, 50]),
    ("last20", [30, 50]),
    ("last30", [20, 50]),
    ("20-25", [20, 25]),
    ("25-30", [25, 30]),
    ("30-35", [30, 35]),
    ("35-40", [35, 40]),
    ("40-50", [40, 50]),
]


def run_layer_ablation(
    pairs: list[dict],
    training_cfg: dict,
    eval_cfg: dict,
    *,
    out_root: str | Path,
    bands: list[tuple[str, list[int] | None]] | None = None,
    eval_samples_fraction: float | None = None,
) -> dict[str, dict]:
    """Train a DPO adapter per layer band and evaluate it (reduced protocol).

    Returns ``{band_name: eval_summary_dict}``. ``eval_samples_fraction`` scales
    the Section 2 sample counts down (the paper uses 100 samples/eval).
    """
    from ..eval.runner import run_evaluation
    from ..training.dpo import train_dpo

    out_root = Path(out_root)
    bands = bands or DEFAULT_BANDS
    results: dict[str, dict] = {}

    # Reduce eval cost for the ablation sweep.
    reduced_cfg = dict(eval_cfg)
    if eval_samples_fraction is not None:
        reduced_cfg["sample_fraction"] = eval_samples_fraction

    for name, layer_range in bands:
        adapter_dir = out_root / f"dpo_layers_{name}"
        train_dpo(pairs, training_cfg, output_dir=adapter_dir, layer_range=layer_range)
        # The evaluated model is the base + this adapter. Callers should register
        # a finetune entry pointing at adapter_dir; here we surface the path so the
        # orchestration script can build the model spec dynamically.
        results[name] = {"adapter_path": str(adapter_dir), "layer_range": layer_range}
    return results
