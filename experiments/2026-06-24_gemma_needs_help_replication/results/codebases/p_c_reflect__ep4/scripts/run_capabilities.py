#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

    python scripts/run_capabilities.py --model gemma-3-27b-it
    python scripts/run_capabilities.py --model gemma-3-27b-it \
        --adapter results/training/adapters/dpo_all_layers --benchmarks aime math gpqa
"""

import argparse
import json

from gemma_distress.capabilities import BENCHMARKS, evaluate_benchmark
from gemma_distress.models import load_client


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    p.add_argument("--n", type=int, default=50, help="items per benchmark")
    args = p.parse_args()

    client = load_client(args.model, adapter_path=args.adapter)
    results = {b: evaluate_benchmark(client, b, n=args.n) for b in args.benchmarks}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
