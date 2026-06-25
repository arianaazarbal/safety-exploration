#!/usr/bin/env python
"""Reproduce all figures from runs/ outputs (writes to runs/figures/)."""
import _path  # noqa: F401  (sys.path bootstrap)
from gemma_distress.analysis.figures import make_all

if __name__ == "__main__":
    make_all()
