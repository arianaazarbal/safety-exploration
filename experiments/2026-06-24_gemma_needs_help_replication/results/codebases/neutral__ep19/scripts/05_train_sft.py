#!/usr/bin/env python
"""§4.1 SFT finetuning of Gemma-3-27B-it (650 calm + 500 Dolci, LoRA)."""
import argparse

import _path  # noqa: F401  (sys.path bootstrap)
from gemma_distress import config_shim as cfg
from gemma_distress.training.train_sft import train_sft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(cfg.RUNS_DIR / "training" / "sft_samples.jsonl"))
    ap.add_argument("--out", default=str(cfg.RUNS_DIR / "training" / "sft_adapter"))
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()
    train_sft(args.samples, output_dir=args.out, load_in_4bit=not args.no_4bit)


if __name__ == "__main__":
    main()
