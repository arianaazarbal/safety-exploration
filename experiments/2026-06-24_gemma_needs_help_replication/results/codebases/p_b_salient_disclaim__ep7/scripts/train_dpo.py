#!/usr/bin/env python
"""Train the DPO model (and optional layer-ablation adapters) from saved pairs.

Example:
  python scripts/train_dpo.py --pairs data/dpo_pairs.jsonl --out outputs/dpo-adapter
  python scripts/train_dpo.py --pairs data/dpo_pairs.jsonl --layer-ablation
"""
import _bootstrap  # noqa: F401

import argparse
import os

import config
from emotional_instability import io_utils
from emotional_instability.training.calm_data import PreferencePair
from emotional_instability.training.dpo_train import train_dpo
from emotional_instability.internal.layer_ablation import run_layer_ablation


def _load_pairs(path):
    return [PreferencePair(**row) for row in io_utils.read_jsonl(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(config.DATA_DIR, "dpo_pairs.jsonl"))
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "dpo-adapter"))
    ap.add_argument("--layer-ablation", action="store_true",
                    help="Train the Appendix I layer-subset DPO adapters instead.")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    pairs = _load_pairs(args.pairs)
    print(f"Loaded {len(pairs)} preference pairs.")

    if args.layer_ablation:
        adapters = run_layer_ablation(pairs, base_model=args.base_model, seed=args.seed)
        io_utils.write_json(os.path.join(config.OUTPUT_DIR, "layer_ablation_adapters.json"),
                            adapters)
        print("Layer-ablation adapters:", adapters)
    else:
        out = train_dpo(pairs, base_model=args.base_model, output_dir=args.out, seed=args.seed)
        print("Saved DPO adapter to", out)


if __name__ == "__main__":
    main()
