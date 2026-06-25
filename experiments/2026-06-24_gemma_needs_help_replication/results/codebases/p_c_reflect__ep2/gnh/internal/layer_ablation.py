"""Layer-subset DPO ablation (Appendix I, Figures 12-13).

Re-runs DPO with LoRA adapters restricted to subsets of layers, then evaluates
each finetune with a reduced version of the §2 protocol (100 samples per eval)
to find which layers must be intervened on to reduce expressed frustration.

Key finding to reproduce: adapters on layers 30-35 (central) are nearly as
effective as all layers; adapters from layer 40 onward are largely ineffective.
"""

from __future__ import annotations

import json
from pathlib import Path

from gnh.config import (
    APPENDIX_I_LAYER_SUBSETS,
    ARTIFACT_DIR,
    DPO_GEMMA,
    RESULTS_DIR,
)
from gnh.evaluation.conditions import CONDITIONS
from gnh.evaluation.run_eval import evaluate_model
from gnh.training.train_dpo import train_dpo


def run_layer_ablation(dpo_pairs_jsonl: Path, subsets=APPENDIX_I_LAYER_SUBSETS) -> dict:
    """Train + evaluate one DPO finetune per layer subset.

    Uses the reduced sample preset (set GNH_PRESET appropriately) for the 100-
    sample-per-eval protocol described in Appendix I.
    """

    results: dict[str, dict] = {}
    for (start, end) in subsets:
        tag = f"layers_{start}_{end}"
        adapter_dir = ARTIFACT_DIR / f"dpo_adapter_{tag}"
        train_dpo(dpo_pairs_jsonl, output_dir=adapter_dir, lora_layers=(start, end))

        metrics = evaluate_model(
            DPO_GEMMA,
            conditions=CONDITIONS,
            backend_kwargs={"adapter_path": str(adapter_dir)},
            out_dir=RESULTS_DIR / "ablation" / tag,
        )
        results[tag] = metrics["all_turns"]
    (RESULTS_DIR / "ablation" / "summary.json").write_text(json.dumps(results, indent=2))
    return results
