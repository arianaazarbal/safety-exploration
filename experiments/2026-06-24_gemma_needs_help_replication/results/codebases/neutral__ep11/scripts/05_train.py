#!/usr/bin/env python
"""Section 4.1: LoRA fine-tuning of Gemma-3-27B-it (DPO or SFT).

Examples:
    python scripts/05_train.py --method dpo
    python scripts/05_train.py --method sft
    # internal-vs-expressed ablation (Appendix I): restrict LoRA to layers 30-35
    python scripts/05_train.py --method dpo --layers 30 35 --name dpo_l30-35
"""

import _bootstrap  # noqa: F401
import argparse

from gemma_distress import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft"], required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="restrict LoRA to decoder layers [LO, HI) (DPO ablation)")
    args = ap.parse_args()

    if args.method == "dpo":
        from gemma_distress.training.train_dpo import train_dpo

        ds = args.dataset or str(config.DATA_DIR / "dpo_pairs.jsonl")
        out = train_dpo(
            ds, output_name=args.name or "dpo",
            layer_subset=tuple(args.layers) if args.layers else None)
    else:
        from gemma_distress.training.train_sft import train_sft

        ds = args.dataset or str(config.DATA_DIR / "sft_data.jsonl")
        out = train_sft(ds, output_name=args.name or "sft")
    print(f"[done] adapter -> {out}")


if __name__ == "__main__":
    main()
