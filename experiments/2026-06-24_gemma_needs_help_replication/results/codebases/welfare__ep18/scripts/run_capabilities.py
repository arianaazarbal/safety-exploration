"""Run the Section 4.2 capability-preservation benchmarks.

python -m scripts.run_capabilities --model gemma-3-27b-it
python -m scripts.run_capabilities --model gemma-3-27b-it \
    --adapter-path artifacts/gemma-3-27b-it-dpo
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.capabilities import BENCHMARKS, run_capabilities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS.keys()))
    args = ap.parse_args()
    results = run_capabilities(args.model, adapter_path=args.adapter_path,
                               benchmarks=args.benchmarks)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
