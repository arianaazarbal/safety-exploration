#!/usr/bin/env python
"""Section 4.1: generate calm finetuning data from Gemma-3-27B-it.

Example
-------
python scripts/03_generate_calm_data.py --variant diverse
python scripts/03_generate_calm_data.py --variant teacher
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.training.generate_calm_data import generate_calm_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    generate_calm_data(variant=args.variant, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
