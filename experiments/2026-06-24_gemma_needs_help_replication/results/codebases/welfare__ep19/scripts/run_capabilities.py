#!/usr/bin/env python
"""Run the capability-preservation evals (Section 4.2, Figure 7).

  python scripts/run_capabilities.py --targets gemma-3-27b-it gemma-3-27b-dpo
  python scripts/run_capabilities.py --benches MATH GPQA --targets gemma-3-27b-dpo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emo_instability.capabilities import run_capabilities
from emo_instability.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--benches", nargs="*", default=None,
                    help="subset: MATH AIME GPQA BBH TruthfulQA (default: all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for t in args.targets:
        run_capabilities(cfg, t, benches=args.benches)


if __name__ == "__main__":
    main()
