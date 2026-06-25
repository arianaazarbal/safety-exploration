#!/usr/bin/env python
"""Section 4.2: recovery experiment (Figure 8).

Truncates high-frustration (>=7) Gemma-3-27B-it responses near their end,
paraphrases, and measures continuations for base / instruct / DPO models.

Example:
  python scripts/10_recovery.py --adapter results/training/dpo_adapter
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from emoinstab.config import get_settings
from emoinstab.prefill.recovery import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="quick", choices=["quick", "full"])
    p.add_argument("--adapter", default=None, help="DPO adapter path")
    args = p.parse_args()

    settings = get_settings(profile=args.profile)
    models = ["gemma-3-27b-pt", "gemma-3-27b-it"]
    if args.adapter:
        models.append("dpo")
    out = run(settings, eval_models=models, dpo_adapter=args.adapter)
    print(f"[recovery] results -> {out}")


if __name__ == "__main__":
    main()
