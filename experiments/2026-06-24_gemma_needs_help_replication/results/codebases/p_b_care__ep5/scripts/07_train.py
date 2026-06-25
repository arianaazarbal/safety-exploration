#!/usr/bin/env python
"""Section 4.1: LoRA finetuning of Gemma-3-27B-it (SFT or DPO).

Usage:
    python scripts/07_train.py dpo
    python scripts/07_train.py sft
    python scripts/07_train.py dpo --layers 30 35      # Appendix I layer ablation
"""
import argparse

import _bootstrap  # noqa: F401  (ensures gemma_distress is importable)
from gemma_distress import config
from gemma_distress.training import train_sft, train_dpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["sft", "dpo"])
    ap.add_argument("--no-4bit", action="store_true",
                    help="disable QLoRA 4-bit base loading (needs more VRAM)")
    ap.add_argument("--layers", nargs=2, type=int, default=None,
                    metavar=("LO", "HI"),
                    help="DPO only: restrict LoRA to decoder layers [LO, HI)")
    args = ap.parse_args()

    load_in_4bit = not args.no_4bit
    if args.method == "sft":
        path = train_sft(str(config.DATA_DIR / "sft_dataset.json"),
                         load_in_4bit=load_in_4bit)
    else:
        layers = tuple(args.layers) if args.layers else None
        out = config.ADAPTER_DIR / ("dpo" if layers is None else f"dpo_L{layers[0]}-{layers[1]}")
        path = train_dpo(str(config.DATA_DIR / "dpo_dataset.json"),
                         output_dir=str(out), load_in_4bit=load_in_4bit, layers=layers)
    print(f"[train] {args.method} adapter saved -> {path}")


if __name__ == "__main__":
    main()
