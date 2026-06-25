#!/usr/bin/env python
"""Section 4.1: generate calm response data from Gemma-3-27B-it with the
reassuring prompt additions, filtered to score-0/1 conversations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from distress_eval.training.generate_calm_data import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=cfg.CALM_DATA_SAMPLES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = generate(n_samples=args.n_samples, seed=args.seed)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
