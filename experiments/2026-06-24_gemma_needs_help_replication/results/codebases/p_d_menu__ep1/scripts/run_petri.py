#!/usr/bin/env python3
"""Run the Petri open-ended emotion elicitation (Section 4.2) for given models."""
from __future__ import annotations

import argparse

from _common import add_common_args, get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["gemma-3-27b-it", "gemma-3-27b-it-dpo",
                                 "gemini-2.5-flash"])
    add_common_args(parser)
    args = parser.parse_args()
    cfg = get_config(args)

    from emotional_instability.petri.run_petri import run_petri

    for model in args.models:
        print(f"\n=== Petri: {model} ===")
        summary = run_petri(cfg, model, out_dir=args.out or "outputs/petri")
        for dim, stats in summary["per_dimension"].items():
            print(f"  {dim:12s} mean={stats['mean']:.2f} "
                  f"CI95=[{stats['ci95'][0]:.2f}, {stats['ci95'][1]:.2f}]")
        print(f"  early_stops={summary['early_stops']} optouts={summary['optouts']}")


if __name__ == "__main__":
    main()
