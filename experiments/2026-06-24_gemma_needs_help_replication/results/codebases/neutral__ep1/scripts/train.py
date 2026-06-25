#!/usr/bin/env python
"""LoRA finetuning of Gemma-3-27B-it (Section 4 / Table 9).

Examples:
    python scripts/train.py dpo --pairs data/dpo_pairs.jsonl
    python scripts/train.py sft --data data/sft_dataset.jsonl
    python scripts/train.py dpo --pairs data/dpo_pairs.jsonl --layers 30 31 32 33 34
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from emostab.config import ADAPTERS_DIR, DATA_DIR
from emostab.training.train import train_dpo, train_sft


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="method", required=True)

    d = sub.add_parser("dpo")
    d.add_argument("--pairs", default=str(DATA_DIR / "dpo_pairs.jsonl"))
    d.add_argument("--out", default=str(ADAPTERS_DIR / "dpo"))
    d.add_argument("--layers", nargs="*", type=int, default=None,
                   help="restrict LoRA to these layers (Appendix I ablation)")
    d.add_argument("--load-in-4bit", action="store_true")

    s = sub.add_parser("sft")
    s.add_argument("--data", default=str(DATA_DIR / "sft_dataset.jsonl"))
    s.add_argument("--out", default=str(ADAPTERS_DIR / "sft_diverse"))
    s.add_argument("--load-in-4bit", action="store_true")

    args = ap.parse_args()
    if args.method == "dpo":
        out = train_dpo(Path(args.pairs), out_dir=Path(args.out),
                        layers=args.layers, load_in_4bit=args.load_in_4bit)
    else:
        out = train_sft(Path(args.data), out_dir=Path(args.out),
                        load_in_4bit=args.load_in_4bit)
    print(f"adapter saved to {out}")


if __name__ == "__main__":
    main()
