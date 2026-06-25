#!/usr/bin/env python
"""Section 4.2 — capability + EmoBench preservation checks (Figure 7).

Compares vanilla Gemma against the fine-tunes on MATH/AIME/GPQA/TruthfulQA/
EmoBench. Reports accuracy per benchmark; the headline result is no regression.

Example
-------
python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo \
    --benchmarks math gpqa truthfulqa emobench --n 100
"""

from __future__ import annotations

import argparse

from emotional_instability.capabilities import BENCHMARKS, run_capabilities
from emotional_instability.models import build_model, load_model_registry


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out-dir", default="outputs/capabilities")
    return p.parse_args()


def main():
    args = parse_args()
    registry = load_model_registry()
    for key in args.models:
        print(f"=== Capabilities: {key} ===")
        model = build_model(key, registry)
        res = run_capabilities(
            model, benchmarks=args.benchmarks, n_per_benchmark=args.n,
            out_path=f"{args.out_dir}/{key}.jsonl",
        )
        for bench, r in res.items():
            print(f"  {bench}: {r}")


if __name__ == "__main__":
    main()
