#!/usr/bin/env python
"""Section 4.1: generate calm data and build SFT + DPO datasets."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_distress.train_data import build_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse",
                    help="calm-data variant (Appendix F). DPO uses 'diverse'.")
    args = ap.parse_args()
    build_datasets(variant=args.variant)


if __name__ == "__main__":
    main()
