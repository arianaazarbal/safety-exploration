#!/usr/bin/env python
"""Aggregate all results into summary CSVs + figures (reproduces Figs 1-3, 6)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_distress.analysis import produce_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="model keys to include in the main-eval summary")
    args = ap.parse_args()
    produce_all(args.models)


if __name__ == "__main__":
    main()
