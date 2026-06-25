#!/usr/bin/env python
"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Evaluates vanilla / DPO / SFT Gemma on math, GPQA, and TruthfulQA (extend with
AIME/BBH/EmoBench per DESIGN.md) and prints accuracies for a side-by-side
no-degradation check.

Example
-------
python scripts/run_capabilities.py --model gemma-3-27b-it \
    --adapter outputs/adapters/dpo --benchmarks math gpqa truthfulqa --n 200
"""

from __future__ import annotations

import argparse

from emotional_instability.capabilities import BENCHMARK_LOADERS, evaluate_benchmark
from emotional_instability.models.registry import load_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=["math", "gpqa", "truthfulqa"])
    ap.add_argument("--n", type=int, default=200, help="items per benchmark")
    args = ap.parse_args()

    backend = load_backend(args.model, adapter_path=args.adapter)
    label = args.model + ("+adapter" if args.adapter else "")
    print(f"=== capabilities: {label} ===")
    for bench in args.benchmarks:
        loader = BENCHMARK_LOADERS.get(bench)
        if loader is None:
            print(f"  {bench}: no loader (see DESIGN.md to add)")
            continue
        items = loader(n=args.n)
        res = evaluate_benchmark(backend, items)
        print(f"  {bench:12s} acc={res['accuracy']:.3f}  (n={res['n']})")
    backend.close()


if __name__ == "__main__":
    main()
