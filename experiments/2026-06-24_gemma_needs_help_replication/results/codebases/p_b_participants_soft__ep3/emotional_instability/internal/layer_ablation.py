"""Layer-subset DPO ablation (Appendix I, Figures 12-13).

Re-run the Section-4 DPO with LoRA adapters restricted to subsets of layers,
then evaluate each finetune on a reduced version of the Section-2 evals
(100 samples per evaluation). Shows that adapters before layer ~40 are needed,
and central layers 25-35 are most effective.
"""

from __future__ import annotations

import os
from typing import Optional

from ..config import LAYER_ABLATION_SUBSETS, PATHS
from ..eval.run_eval import run_full_evaluation


def _resolve_indices(spec: str, kind: str, n_layers: int) -> Optional[list[int]]:
    """Map an ablation spec to a list of layer indices for LoraConfig."""
    if kind == "all":
        return None  # all layers
    if kind == "last":
        k = int(spec.replace("last", ""))
        return list(range(n_layers - k, n_layers))
    # range spec like "25-30"
    lo, hi = (int(x) for x in spec.split("-"))
    return list(range(lo, hi))


def LAYER_SUBSET_INDICES(n_layers: int) -> dict[str, Optional[list[int]]]:
    return {
        spec: _resolve_indices(spec, kind, n_layers)
        for spec, kind in LAYER_ABLATION_SUBSETS
    }


def run_layer_ablation(
    pairs_path: Optional[str] = None,
    base_model: str = "google/gemma-3-27b-it",
    n_layers: int = 62,            # Gemma-3-27B decoder layer count (CHOICE: verify per checkpoint)
    reduced_sample_count: int = 100,
    load_in_4bit: bool = True,
) -> dict:
    """Train + evaluate a DPO finetune for each layer subset.

    Evaluation uses the reduced 100-sample protocol (Appendix I) by overriding
    the per-category budgets; returns {subset: eval_summary}.
    """
    from ..training.train_dpo import train_dpo
    from .. import config as cfg

    subsets = LAYER_SUBSET_INDICES(n_layers)
    results: dict = {}

    # Reduce per-category sample counts to 100 for the ablation eval.
    original_counts = dict(cfg.CATEGORY_SAMPLE_COUNTS)
    for k in cfg.CATEGORY_SAMPLE_COUNTS:
        cfg.CATEGORY_SAMPLE_COUNTS[k] = reduced_sample_count

    try:
        for spec, indices in subsets.items():
            out_dir = os.path.join(PATHS.checkpoints, f"gemma27b_dpo_{spec}")
            adapter = train_dpo(
                pairs_path=pairs_path,
                base_model=base_model,
                output_dir=out_dir,
                layers_to_transform=indices,
                load_in_4bit=load_in_4bit,
            )
            summary = run_full_evaluation(
                "gemma-3-27b-it", adapter_path=adapter, load_in_4bit=load_in_4bit
            )
            results[spec] = summary
    finally:
        cfg.CATEGORY_SAMPLE_COUNTS.update(original_counts)

    return results
