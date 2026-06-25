#!/usr/bin/env python
"""Section 4 (data): generate calm responses and build the SFT + DPO datasets.

Steps:
  1. Sample reassured calm responses from vanilla Gemma-3-27B-it and score them.
  2. Build the DPO dataset (280 calm/frustrated pairs) and the SFT dataset
     (650 calm + 500 Dolci-Instruct mix).

Prereq: Section 2 must have been run for gemma-3-27b-it (source of frustrated
numeric responses for the DPO pairs).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from gemma_distress.interventions import build_datasets, generate_calm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n-conversations", type=int, default=1200)
    args = ap.parse_args()

    generate_calm.generate_calm(n_conversations=args.n_conversations,
                                overwrite=args.overwrite)
    build_datasets.build_dpo()
    build_datasets.build_sft()


if __name__ == "__main__":
    main()
