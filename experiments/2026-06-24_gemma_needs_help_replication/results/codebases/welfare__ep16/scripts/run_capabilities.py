#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks for one or more models."""
import argparse

from gemma_distress import config
from gemma_distress.capabilities import evaluate_capabilities
from gemma_distress.models import build_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "dpo-gemma"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n-per-bench", type=int, default=50)
    args = ap.parse_args()

    for model_key in args.models:
        client = build_client(model_key, adapter_path=args.adapter)
        summary = evaluate_capabilities(model_key, client, n_per_bench=args.n_per_bench)
        print(f"[capabilities] {model_key}: {summary}")


if __name__ == "__main__":
    main()
