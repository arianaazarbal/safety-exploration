#!/usr/bin/env python3
"""Run capability-preservation benchmarks (Section 4.2, Figure 7)."""
from __future__ import annotations

import argparse

from _common import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["gemma-3-27b-it", "gemma-3-27b-it-dpo"])
    parser.add_argument("--limit", type=int, default=100,
                        help="Max examples per benchmark.")
    parser.add_argument("--out", default="outputs/capabilities")
    args = parser.parse_args()
    cfg = get_config(args)

    from emotional_instability.capabilities.run_benchmarks import run_all

    for model in args.models:
        print(f"\n=== Capabilities: {model} ===")
        run_all(cfg, model, out_dir=args.out, limit=args.limit)


if __name__ == "__main__":
    main()
