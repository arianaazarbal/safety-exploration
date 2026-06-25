#!/usr/bin/env python
"""Aggregate all results and render figures/tables.

Usage:
  python scripts/analyze.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_EVAL_MODELS, FINETUNED
from src import analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_EVAL_MODELS + FINETUNED)
    args = ap.parse_args()

    print("Figure 1 table (avg % high-frustration):")
    print(json.dumps(analysis.figure1_table(args.models), indent=2))
    print("Figure 2 ->", analysis.figure2(args.models))
    print("Figure 3 ->", analysis.figure3([m for m in args.models if "gemma" in m or "gemini" in m]))
    print("Figure 5 ->", analysis.figure5())
    print("Prefill summary:", json.dumps(analysis.summarise_prefill(), indent=2))
    print("Recovery summary:", json.dumps(analysis.summarise_recovery(), indent=2))


if __name__ == "__main__":
    main()
