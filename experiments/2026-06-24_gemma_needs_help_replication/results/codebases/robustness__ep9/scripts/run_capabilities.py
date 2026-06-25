#!/usr/bin/env python
"""Capability-preservation benchmarks (Section 4.2).

Compare a base/instruct model against a finetuned adapter to confirm no
capability regression.

  python scripts/run_capabilities.py --model gemma-3-27b-it --tag vanilla
  python scripts/run_capabilities.py --model gemma-3-27b-it --adapter outputs/dpo --tag dpo
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os

from emo_instability.capabilities import BENCHMARKS, run_capability_suite
from emo_instability.models import build_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="vanilla")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--out", default="outputs/capabilities")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    client = build_client(args.model, adapter_path=args.adapter)
    results = run_capability_suite(client, benchmarks=args.benchmarks, n_samples=args.n_samples)

    accs = {name: r["accuracy"] for name, r in results.items()}
    print(json.dumps(accs, indent=2))
    with open(os.path.join(args.out, f"capabilities_{args.tag}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
