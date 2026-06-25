#!/usr/bin/env python
"""Aggregate all run files into metrics + figures.

    python scripts/make_figures.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import aggregate, plots


def main():
    summary = aggregate.write_summary()
    print(json.dumps(summary.get("headline_avg_high", []), indent=2))
    plots.make_all()


if __name__ == "__main__":
    main()
