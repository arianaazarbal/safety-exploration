#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion probe. Compares vanilla Gemma vs a
DPO adapter on the same frustrated conversation (Figures 14/15).

Example
-------
python scripts/10_internal_emotions.py --vanilla gemma-3-27b-it \
    --dpo-adapter artifacts/gemma-3-27b-it-dpo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.internal_emotions import compare_models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla", default="gemma-3-27b-it")
    ap.add_argument("--dpo-adapter", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = compare_models(args.vanilla, args.dpo_adapter, seed=args.seed)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
