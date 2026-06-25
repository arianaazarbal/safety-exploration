#!/usr/bin/env python
"""Section 4.2 / Figure 7: capability-preservation benchmarks for the vanilla
and finetuned Gemma models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.capabilities import run_capability_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it"])
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--n", type=int, default=100,
                    help="items per benchmark (paper uses subsets)")
    args = ap.parse_args()

    models = list(args.models)
    if args.dpo_adapter:
        config.register_lora_variant("gemma-3-27b-dpo", "gemma-3-27b-it",
                                     args.dpo_adapter, display="DPO Gemma")
        models.append("gemma-3-27b-dpo")
    if args.sft_adapter:
        config.register_lora_variant("gemma-3-27b-sft", "gemma-3-27b-it",
                                     args.sft_adapter, display="SFT Gemma")
        models.append("gemma-3-27b-sft")

    path = run_capability_suite(models, n_per_benchmark=args.n)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
