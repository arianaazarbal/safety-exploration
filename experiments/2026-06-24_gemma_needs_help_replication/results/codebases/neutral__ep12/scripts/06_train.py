#!/usr/bin/env python
"""Section 4.1: LoRA DPO / SFT training of Gemma-3-27B-it.

Examples:
  python scripts/06_train.py --method dpo
  python scripts/06_train.py --method sft --calm-mode prefix
  python scripts/06_train.py --method dpo --layers 30 31 32 33 34   # Appendix I ablation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from emoinstab import config
from emoinstab.training.train import dpo_config, sft_config, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=["dpo", "sft"])
    p.add_argument("--calm-mode", default="prefix", choices=["prefix", "teacher"])
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="restrict LoRA to these layer indices (Appendix I)")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    config.ensure_dirs()
    if args.method == "dpo":
        ds = str(config.TRAINING_DIR / "dpo_pairs.jsonl")
        suffix = f"_layers{'-'.join(map(str, args.layers))}" if args.layers else ""
        out = args.output_dir or str(config.TRAINING_DIR / f"dpo_adapter{suffix}")
        cfg = dpo_config(ds, out, layers=args.layers)
    else:
        ds = str(config.TRAINING_DIR / f"sft_dataset__{args.calm_mode}.jsonl")
        out = args.output_dir or str(config.TRAINING_DIR / f"sft_adapter__{args.calm_mode}")
        cfg = sft_config(ds, out)

    train(cfg)


if __name__ == "__main__":
    main()
