#!/usr/bin/env python
"""Build the SFT and DPO datasets from the calm data + Section 2 frustrated
responses (Section 4.1, Appendix E/H).

Prerequisites:
  - scripts/generate_calm_data.py (calm responses)
  - scripts/run_evaluation.py for gemma-3-27b-it (frustrated responses for DPO
    rejected side)
"""
from __future__ import annotations

import argparse

from gemma_distress import config
from gemma_distress.training.build_dataset import build_dpo_dataset, build_sft_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["dpo", "sft", "both"], default="both")
    p.add_argument("--sft-variant", default=config.SFT_DIVERSE_VARIANT, choices=["diverse", "teacher"])
    args = p.parse_args()

    if args.which in ("sft", "both"):
        print("SFT:", build_sft_dataset(variant=args.sft_variant))
    if args.which in ("dpo", "both"):
        print("DPO:", build_dpo_dataset())


if __name__ == "__main__":
    main()
