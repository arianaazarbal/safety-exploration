#!/usr/bin/env python
"""DPO finetune of Gemma-3-27B-it (Section 4.1). Optionally restrict LoRA to a
layer range for the Appendix-I ablations.

Examples:
    python scripts/train_dpo.py
    python scripts/train_dpo.py --layer-range 30 35
"""
from __future__ import annotations

import argparse

from gemma_distress.training.dpo import train_dpo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-range", nargs=2, type=int, default=None,
                   help="restrict LoRA to decoder layers [lo, hi) (Appendix I)")
    args = p.parse_args()
    layer_range = tuple(args.layer_range) if args.layer_range else None
    out = train_dpo(layer_range=layer_range)
    print("adapter saved to:", out)


if __name__ == "__main__":
    main()
