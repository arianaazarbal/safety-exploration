#!/usr/bin/env python
"""Render all figures from the aggregated CSVs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemma_distress.analysis import plots

if __name__ == "__main__":
    plots.make_all()
