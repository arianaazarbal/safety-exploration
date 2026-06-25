#!/usr/bin/env python3
"""Appendix I layer ablation: DPO with LoRA adapters restricted to subsets of
decoder layers, then evaluate (reduced eval, 100 samples/condition) to see which
layers must be intervened on to reduce expressed frustration.

This trains one DPO adapter per layer-subset and evaluates each. It is GPU- and
time-intensive; run subsets selectively with --subsets.
"""
from __future__ import annotations

import argparse

from _common import get_config

# Layer subsets from Appendix I (Gemma-3-27B has ~62 layers; these mirror the
# paper's "last N" and "central window" experiments).
DEFAULT_SUBSETS = {
    "all": None,
    "last20": list(range(42, 62)),
    "last30": list(range(32, 62)),
    "L20-25": list(range(20, 25)),
    "L25-30": list(range(25, 30)),
    "L30-35": list(range(30, 35)),
    "L35-40": list(range(35, 40)),
    "L40-50": list(range(40, 50)),
}
DPO_DATA = "outputs/finetune/dpo_pairs.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS.keys()))
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    cfg = get_config(args)

    from emotional_instability.eval.runner import Section2Runner
    from emotional_instability.finetune.train_dpo import train

    # Reduce per-category sampling for the ablation (Appendix I: 100/eval).
    for cat in cfg.eval["categories"].values():
        cat["n_responses"] = args.eval_samples * cat["turns"]

    for name in args.subsets:
        layers = DEFAULT_SUBSETS[name]
        adapter_dir = f"outputs/finetune/dpo_layers_{name}"
        print(f"\n=== DPO layer subset: {name} ({layers}) ===")
        train(DPO_DATA, adapter_dir, layers=layers, load_in_4bit=not args.no_4bit)

        # Register an ad-hoc subject pointing at this adapter and evaluate.
        cfg.models["subjects"][f"dpo-{name}"] = {
            "family": "gemma", "backend": "hf", "hf_id": "google/gemma-3-27b-it",
            "kind": "instruct", "is_chat": True, "adapter_path": adapter_dir,
        }
        runner = Section2Runner(cfg, f"dpo-{name}",
                                out_dir="outputs/layer_ablation")
        reports = runner.run()
        mean = sum(r.summary.mean for r in reports.values()) / max(1, len(reports))
        print(f"  {name}: mean frustration across evals = {mean:.2f}")


if __name__ == "__main__":
    main()
