#!/usr/bin/env python3
"""Aggregate judged scores into Figure 1/2/3 metrics (CSV + optional PNG).

    python scripts/analyze.py --all [--plots]
    python scripts/analyze.py --models gemma-3-27b-it gemini-2.5-flash
"""
import _bootstrap  # noqa: F401
import argparse

from distress_eval.analyze import run_analysis
from distress_eval.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--plots", action="store_true", help="also render PNG figures")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    models = list(cfg.targets) if args.all else args.models
    if not models:
        ap.error("specify --models <names...> or --all")

    results_dir = cfg.paths.resolve("results_dir")
    out_dir = results_dir / "analysis"
    run_analysis(results_dir, models, out_dir, plots=args.plots)
    print(f"\nWrote analysis CSVs to {out_dir}")


if __name__ == "__main__":
    main()
