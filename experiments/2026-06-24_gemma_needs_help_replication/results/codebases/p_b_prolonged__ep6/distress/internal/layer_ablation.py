"""Layer-ablation DPO experiments (Appendix I, Figures 12-13).

Trains DPO with LoRA adapters restricted to subsets of decoder layers, then
evaluates each with the reduced (100-sample) version of the Section 2 eval, to
test the claim that intervening on central layers (25-35) is necessary -- i.e.
the intervention acts on internal states, not just final-layer expression.

The layer ranges below mirror the paper:
  * cumulative-from-end: last 5, last 10, ..., all layers
  * central bands: 20-25, 25-30, 30-35, 35-40, 40-50
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import CHECKPOINT_DIR, INTERVENTION_BASE, RESULTS_DIR, get_model
from ..eval.conditions import DEFAULT_CONDITIONS
from ..eval.runner import run_eval
from ..models import build_finetuned_client
from ..training.dpo_train import train_dpo

# Gemma-3-27B has 62 decoder layers; the paper indexes "last N" from the end and
# central bands by absolute index. We expose both.
N_LAYERS_27B = 62

CENTRAL_BANDS = {
    "20-25": list(range(20, 25)),
    "25-30": list(range(25, 30)),
    "30-35": list(range(30, 35)),
    "35-40": list(range(35, 40)),
    "40-50": list(range(40, 50)),
}


def cumulative_from_end(n_layers: int = N_LAYERS_27B) -> dict[str, list[int]]:
    out = {}
    for last in (5, 10, 20, 30, n_layers):
        out[f"last-{last}"] = list(range(max(0, n_layers - last), n_layers))
    return out


@dataclass
class AblationResult:
    label: str
    layers: list[int]
    adapter_dir: str
    eval_path: str


def run_layer_ablations(
    *,
    bands: Optional[dict[str, list[int]]] = None,
    fraction: float = 100 / 4000,   # ~100 samples per evaluation (Appendix I)
    seed: int = 0,
) -> list[AblationResult]:
    """Train + evaluate a DPO adapter for each layer band. Returns metadata.

    Note: this trains many models and is the most expensive experiment; intended
    to be launched piecewise. `fraction` reduces the eval to ~100 samples per
    condition as in the paper's reduced runs.
    """
    bands = bands or {**cumulative_from_end(), **CENTRAL_BANDS}
    results = []
    for label, layers in bands.items():
        adapter_dir = CHECKPOINT_DIR / f"gemma27b-dpo-layers-{label}"
        train_dpo(output_dir=adapter_dir, lora_layers=layers, seed=seed)
        client = build_finetuned_client(INTERVENTION_BASE, str(adapter_dir))
        eval_path = run_eval(
            INTERVENTION_BASE, client=client, fraction=fraction,
            tag=f"dpo-layers-{label}", seed=seed)
        results.append(AblationResult(label, layers, str(adapter_dir),
                                      str(eval_path)))
    summary = RESULTS_DIR / "layer_ablation_summary.json"
    summary.write_text(json.dumps([vars(r) for r in results], indent=2))
    return results
