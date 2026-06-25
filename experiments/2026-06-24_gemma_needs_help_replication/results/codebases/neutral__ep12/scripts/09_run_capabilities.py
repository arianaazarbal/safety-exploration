#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Evaluates vanilla Gemma-3-27B-it and (optionally) a finetuned adapter on
AIME / MATH / GPQA / BBH / TruthfulQA / EmoBench.

Examples:
  python scripts/09_run_capabilities.py --tag vanilla
  python scripts/09_run_capabilities.py --tag dpo --adapter results/training/dpo_adapter
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from emoinstab.capabilities.benchmarks import BENCHMARKS, evaluate
from emoinstab.config import get_settings
from emoinstab.models.factory import build_client


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    p.add_argument("--profile", default="quick", choices=["quick", "full"])
    args = p.parse_args()

    settings = get_settings(profile=args.profile)
    model = build_client("gemma-3-27b-it", settings, adapter_path=args.adapter)
    out = evaluate(model, settings, args.benchmarks, tag=args.tag)
    print(f"[capabilities] results -> {out}")


if __name__ == "__main__":
    main()
