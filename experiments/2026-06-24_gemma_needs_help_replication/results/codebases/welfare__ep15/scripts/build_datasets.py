#!/usr/bin/env python
"""Section 4.1: build the DPO (280 pairs) and SFT (1150 sample) datasets
from the generated pools.

    python scripts/build_datasets.py            # both
    python scripts/build_datasets.py --only dpo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotional_instability.dpo.build_dataset import build_dpo_dataset, build_sft_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["dpo", "sft", "sft_teacher"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.only in (None, "dpo"):
        build_dpo_dataset(seed=args.seed)
    if args.only in (None, "sft"):
        build_sft_dataset(seed=args.seed, teacher=False)
    if args.only == "sft_teacher":
        build_sft_dataset(seed=args.seed, teacher=True)


if __name__ == "__main__":
    main()
