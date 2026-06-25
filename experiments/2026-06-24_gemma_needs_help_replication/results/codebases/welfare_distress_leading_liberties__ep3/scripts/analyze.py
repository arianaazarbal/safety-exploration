#!/usr/bin/env python3
"""Aggregate judged turns into the paper's metrics and write CSVs.

Usage:
    python scripts/analyze.py --config config.yaml [--cross-validate]

Writes tables under <out_dir>/metrics/ and prints the headline table. With
--cross-validate (and cross_validation.enabled in the config), also re-scores a
random subset with the secondary judge and reports Pearson r / within-1-point.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distress_eval import analysis
from distress_eval.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--cross-validate", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    analysis.compute_metrics(cfg)
    if args.cross_validate:
        analysis.cross_validate(cfg)


if __name__ == "__main__":
    main()
