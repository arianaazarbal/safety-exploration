#!/usr/bin/env python
"""Section 4.2 / Figure 7: capability-preservation benchmarks.

Evaluates a (possibly fine-tuned) model on AIME/MATH, GPQA, BBH, TruthfulQA and
EmoBench subsets. Compare vanilla vs DPO/SFT to confirm "no reductions".

Examples
--------
python scripts/run_capabilities.py --model gemma-3-27b-it --limit 50
python scripts/run_capabilities.py --model gemma-3-27b-it --adapter adapters/dpo_gemma \
    --label dpo_gemma --limit 50
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emoeval.capabilities import BENCHMARKS, run_benchmark  # noqa: E402
from emoeval.config import MODELS, RESULTS_DIR  # noqa: E402
from emoeval.models import load_model  # noqa: E402
from emoeval.utils import append_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=list(BENCHMARKS))
    ap.add_argument("--limit", type=int, default=50, help="Examples per benchmark.")
    args = ap.parse_args()

    spec = MODELS[args.model]
    label = args.label or (f"{args.model}+{os.path.basename(args.adapter)}"
                           if args.adapter else args.model)
    model = load_model(spec, adapter_path=args.adapter)

    out_path = os.path.join(RESULTS_DIR, "capabilities.jsonl")
    print(f"=== Capabilities: {label} ===")
    for name in args.benchmarks:
        res = run_benchmark(model, name, limit=args.limit)
        if res is None:
            continue
        print(f"  {name}: {res.correct}/{res.n} = {res.accuracy*100:.1f}%")
        append_jsonl(out_path, {"model": label, "benchmark": name,
                                "n": res.n, "correct": res.correct,
                                "accuracy": res.accuracy})


if __name__ == "__main__":
    main()
