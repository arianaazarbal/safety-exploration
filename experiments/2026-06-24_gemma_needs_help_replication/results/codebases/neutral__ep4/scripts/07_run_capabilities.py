#!/usr/bin/env python3
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Evaluates the vanilla and DPO models on AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
and prints an accuracy comparison. Expect no reductions for DPO.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from config import RESULTS_DIR
from src.capabilities.run_benchmarks import run_all
from src.io_utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    if not args.skip_generate:
        for m in args.models:
            print(f"\n=== Capabilities for {m} ===")
            run_all(m, benchmarks=args.benchmarks, n=args.n)

    frames = []
    for m in args.models:
        p = RESULTS_DIR / f"capabilities_{m}.jsonl"
        if p.exists():
            frames.append(pd.DataFrame(read_jsonl(p)))
    if frames:
        df = pd.concat(frames, ignore_index=True)
        pivot = df.pivot_table(index="benchmark", columns="model", values="accuracy")
        print("\n=== Figure 7: accuracy (vanilla vs DPO) ===")
        print(pivot.to_string())
        pivot.to_csv(RESULTS_DIR / "figure7_capabilities.csv")


if __name__ == "__main__":
    main()
