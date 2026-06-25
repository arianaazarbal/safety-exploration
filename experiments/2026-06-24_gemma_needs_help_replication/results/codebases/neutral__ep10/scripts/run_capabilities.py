#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Compares vanilla Gemma-3-27b-it against a finetune (via --adapter) on
AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench. The paper finds no reduction in scores.

Examples:
    python scripts/run_capabilities.py --model gemma-3-27b-it
    python scripts/run_capabilities.py --model gemma-3-27b-it --adapter checkpoints/gemma27b_dpo
"""

from __future__ import annotations

import argparse
import json
import os

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from emotional_instability import config
from emotional_instability.capabilities import benchmarks
from emotional_instability.models.registry import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.TARGET_FINETUNE_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=None,
                    choices=list(benchmarks.ALL_BENCHMARKS))
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "capabilities"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.model, adapter_path=args.adapter)
    results = benchmarks.run_all(model, which=args.benchmarks)

    tag = args.model + ("_dpo" if args.adapter else "_vanilla")
    payload = {r.name: {"accuracy": r.accuracy, "n": r.n} for r in results}
    with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
