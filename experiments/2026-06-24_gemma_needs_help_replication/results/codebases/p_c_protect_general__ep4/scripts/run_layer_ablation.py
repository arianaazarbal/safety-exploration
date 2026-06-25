#!/usr/bin/env python
"""Train the Appendix I layer-ablation DPO finetunes (LoRA on layer subsets).

Trains each ablation adapter; evaluate them afterwards with
run_section4_eval.py using a reduced sample count (--n 100, per Appendix I).
"""
import _bootstrap  # noqa: F401
import argparse

from emotional_instability.internal.layer_ablation import all_layer_ablation_configs
from emotional_instability.training.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-dataset", default="data/dpo_dataset.jsonl")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of ablation names to train")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cfgs = all_layer_ablation_configs(args.dpo_dataset)
    for name, cfg in cfgs.items():
        if args.only and name not in args.only:
            continue
        cfg.load_in_4bit = args.load_in_4bit
        print(f"Training {name}: layers={cfg.layers_to_transform}")
        out = train(cfg)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
