#!/usr/bin/env python3
"""Capability-preservation benchmarks (Figure 7): vanilla vs DPO Gemma.

Example:
    python scripts/run_capability.py --model gemma-3-27b-it
    python scripts/run_capability.py --model gemma-3-27b-it --adapter runs/models/dpo
"""

import argparse
import json

import _bootstrap  # noqa: F401
from emotional_instability.config import load_config
from emotional_instability.capability.runner import run_capability


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = run_capability(cfg, args.model, adapter_path=args.adapter, benchmarks=args.benchmarks)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
