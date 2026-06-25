#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

    python scripts/run_capabilities.py --models gemma-3-27b-it
    python scripts/run_capabilities.py --models dpo-gemma-3-27b --lora results/section4/adapters/dpo
    python scripts/run_capabilities.py --models gemma-3-27b-it --benchmarks math gpqa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.capabilities.run import run_capabilities


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=list(config.CAPABILITY_BENCHMARKS))
    args = ap.parse_args()
    for m in args.models:
        run_capabilities(m, lora_path=args.lora, benchmarks=args.benchmarks)


if __name__ == "__main__":
    main()
