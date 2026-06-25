"""Appendix I: LoRA layer-subset DPO ablation.

Re-runs DPO restricting LoRA adapters to subsets of decoder layers, then
evaluates each finetune with a reduced Section-2 protocol (100 samples per
eval), to reproduce the finding that adapters on central layers (~25-35) are
nearly as effective as all-layer DPO while late layers (40+) are not.

Example:
    python -m emotional_instability.scripts.run_layer_ablation \
        --dpo-data data/dpo/dpo_pairs.jsonl --subsets last20 last30 l30_35 l40_50
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from ..config import load_config
from ..eval.metrics import category_summary
from ..eval.runner import run_section2_for_model
from ..training.train_dpo import _resolve_layer_range, train_dpo
from ..utils.io import read_jsonl


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpo-data", required=True)
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="names from eval_config internal.layer_ablation_subsets")
    ap.add_argument("--reduced-n", type=int, default=100,
                    help="samples per eval category for the reduced protocol")
    args = ap.parse_args()

    # Gemma-3-27B has 62 layers; resolve ranges against the real count at train.
    subsets = {s["name"]: s["layers"]
               for s in cfg.eval["internal"]["layer_ablation_subsets"]}
    chosen = args.subsets or list(subsets)

    # Reduce sample counts in-memory for the ablation eval.
    for cat in cfg.eval["categories"].values():
        cat["target_responses"] = min(cat["target_responses"], args.reduced_n)

    results = {}
    n_layers = 62
    for name in chosen:
        layers = _resolve_layer_range(subsets[name], n_layers)
        out_dir = cfg.path("outputs_dir") / "internal" / f"dpo_{name}"
        adapter = train_dpo(dataset_path=args.dpo_data, output_dir=out_dir,
                            layers_to_transform=layers, cfg=cfg)
        # Register the adapter under a transient model name. ModelSpec is a
        # frozen dataclass, so build a fresh copy (do NOT mutate the shared
        # gemma-3-27b-it-dpo spec by reference).
        model_name = f"gemma-3-27b-it-dpo-{name}"
        cfg.participants[model_name] = replace(
            cfg.model("gemma-3-27b-it-dpo"),
            name=model_name, lora_adapter=str(adapter),
        )
        path = run_section2_for_model(model_name, cfg=cfg, overwrite=True)
        summary = category_summary(list(read_jsonl(path)))
        means = [v["mean"] for v in summary.get(model_name, {}).values()]
        results[name] = {"mean_frustration": sum(means) / len(means) if means else None}

    out = cfg.path("outputs_dir") / "internal" / "layer_ablation.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
