#!/usr/bin/env python
"""Section 4.1: build DPO preference pairs and SFT data from generated rollouts.

Example:
    python scripts/04_build_datasets.py --method dpo
    python scripts/04_build_datasets.py --method sft
"""

import _bootstrap  # noqa: F401
import argparse

from transformers import AutoTokenizer

from gemma_distress import config
from gemma_distress.training.datasets import build_dpo_dataset, build_sft_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "sft", "both"], default="both")
    ap.add_argument("--calm", default=str(config.DATA_DIR / "calm_diverse_rollouts.jsonl"))
    ap.add_argument("--frustrated", default=str(config.DATA_DIR / "frustrated_rollouts.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(config.GEMMA_27B_IT.model_id)
    from pathlib import Path

    if args.method in ("dpo", "both"):
        out = build_dpo_dataset(Path(args.calm), Path(args.frustrated),
                                tokenizer, seed=args.seed)
        print(f"[done] DPO dataset -> {out}")
    if args.method in ("sft", "both"):
        out = build_sft_dataset(Path(args.calm), tokenizer, seed=args.seed)
        print(f"[done] SFT dataset -> {out}")


if __name__ == "__main__":
    main()
