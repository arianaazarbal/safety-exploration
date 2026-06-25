#!/usr/bin/env python
"""Generate figures/tables from saved runs.

Example:
  python scripts/make_figures.py --eval-glob 'runs/eval/*/responses.jsonl'
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from emotional_instability import figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-glob", default="runs/eval/*/responses.jsonl")
    ap.add_argument("--finetune-glob", default="runs/eval/*-dpo/responses.jsonl")
    ap.add_argument("--word-models", nargs="*", default=["gemma-3-27b-it", "gemini-2.5-flash"])
    args = ap.parse_args()

    paths = glob.glob(args.eval_glob)
    if not paths:
        print(f"No eval files match {args.eval_glob}")
        return

    fig1 = figures.figure1_table(paths)
    print("Figure 1 (avg % high-frustration):")
    print(fig1.to_string(index=False))
    figures.figure2(paths)
    figures.figure3(paths)
    tbl = figures.table3(paths, args.word_models)
    print("\nTable 3 (differential words):")
    print(tbl.to_string(index=False))

    ft = glob.glob(args.finetune_glob)
    if ft:
        figures.figure5(paths, ft)
    print(f"\nFigures written under runs/figures/")


if __name__ == "__main__":
    main()
