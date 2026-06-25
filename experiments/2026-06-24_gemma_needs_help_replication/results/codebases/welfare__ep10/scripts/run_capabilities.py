#!/usr/bin/env python
"""Run capability-preservation benchmarks (Section 4.2 / Figure 7).

Compares vanilla Gemma-it against the DPO/SFT models on AIME, MATH, GPQA, BBH,
TruthfulQA, and EmoBench, confirming the intervention does not degrade
capabilities.

Examples:
    python -m scripts.run_capabilities --models gemma-3-27b-it gemma-3-27b-dpo
    python -m scripts.run_capabilities --models gemma-3-27b-it --benchmarks math gpqa --limit 20
"""

from __future__ import annotations

import argparse
import json

from capabilities import benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="*", default=None,
                    choices=list(benchmarks.BENCHMARKS))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per benchmark (quick runs)")
    args = ap.parse_args()

    results = benchmarks.run_all(args.models, args.benchmarks, limit=args.limit)
    print(json.dumps(results, indent=2))

    # Pretty per-model table.
    by_model = {}
    for r in results:
        by_model.setdefault(r["model_key"], {})[r["benchmark"]] = r["accuracy"]
    print("\n--- Accuracy by model ---")
    for mk, accs in by_model.items():
        line = ", ".join(f"{b}={a:.3f}" for b, a in accs.items())
        print(f"{mk}: {line}")


if __name__ == "__main__":
    main()
