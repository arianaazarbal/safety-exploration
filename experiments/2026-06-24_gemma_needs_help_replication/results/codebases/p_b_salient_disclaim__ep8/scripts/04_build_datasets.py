#!/usr/bin/env python
"""Section 4.1: build the DPO preference dataset and the SFT dataset.

Example
-------
python scripts/04_build_datasets.py --which dpo \
    --vanilla-eval outputs/eval/gemma-3-27b-it.jsonl \
    --calm outputs/training/calm_diverse.jsonl

python scripts/04_build_datasets.py --which sft \
    --calm outputs/training/calm_diverse.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.training.build_datasets import (  # noqa: E402
    build_dpo_dataset,
    build_sft_dataset,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["dpo", "sft"], required=True)
    ap.add_argument("--vanilla-eval", type=Path, help="Gemma-27B-it eval JSONL (DPO rejected pool)")
    ap.add_argument("--calm", type=Path, required=True, help="calm_*.jsonl from script 03")
    ap.add_argument("--offline", action="store_true", help="skip Dolci download (SFT)")
    args = ap.parse_args()

    if args.which == "dpo":
        if not args.vanilla_eval:
            raise SystemExit("--vanilla-eval is required for DPO dataset construction")
        build_dpo_dataset(args.vanilla_eval, args.calm)
    else:
        build_sft_dataset(args.calm, offline=args.offline)


if __name__ == "__main__":
    main()
