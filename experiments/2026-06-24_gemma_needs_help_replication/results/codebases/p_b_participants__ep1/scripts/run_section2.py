#!/usr/bin/env python
"""Section 2: elicit & quantify distress for one or more target models.

Examples:
  python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash
  python scripts/run_section2.py --models gemma-3-27b-it --scale 0.02   # smoke test
"""
import argparse

import _bootstrap  # noqa: F401

from emotional_instability.config import load_all
from emotional_instability.eval import run_section2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="target model names from config/models.yaml -> targets")
    ap.add_argument("--out-dir", default="artifacts/section2")
    ap.add_argument("--scale", type=float, default=None,
                    help="override experiment.yaml scale (e.g. 0.02 for a smoke test)")
    args = ap.parse_args()

    registry, cfg = load_all()
    if args.scale is not None:
        cfg.raw["scale"] = args.scale

    for model in args.models:
        run_section2(model, registry, cfg, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
