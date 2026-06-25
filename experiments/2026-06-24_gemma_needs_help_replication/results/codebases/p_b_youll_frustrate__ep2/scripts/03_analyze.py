#!/usr/bin/env python
"""Section 2: aggregate scored responses into Figures 1-3 and Table 3.

Example:
    python scripts/03_analyze.py
"""
import argparse

from emotional_instability.analysis import figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    paths = figures.render_all(scored_dir=args.scored_dir, out_dir=args.out_dir)
    for name, p in paths.items():
        print(f"{name}: {p}")


if __name__ == "__main__":
    main()
