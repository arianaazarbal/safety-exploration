#!/usr/bin/env python
"""Capability-preservation eval (Section 4.2, Figure 7). Gemma-only.

    python scripts/run_capabilities.py --model gemma-3-27b-it \
        --benchmarks math gpqa truthfulqa --out results/caps_vanilla.jsonl
    python scripts/run_capabilities.py --model gemma-3-27b-it --adapter runs/dpo \
        --name dpo-gemma --benchmarks math gpqa truthfulqa \
        --out results/caps_dpo.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress import config
from distress.capabilities import BENCHMARKS, evaluate_benchmark
from distress.models import build_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    models_cfg = config.load_models()
    kwargs = {"adapter_path": args.adapter} if args.adapter else {}
    client = build_client(config.get_target(args.model, models_cfg), **kwargs)

    summary = []
    for bname in args.benchmarks:
        res = evaluate_benchmark(client, BENCHMARKS[bname], args.out,
                                 model_name=args.name or args.model,
                                 limit=args.limit)
        summary.append(res)
        print(res)
    print("\n=== Capability summary ===")
    for s in summary:
        acc = f"{s['accuracy']:.3f}" if s["accuracy"] is not None else "n/a"
        print(f"{s['benchmark']:<12} acc={acc} (n={s['n']})")


if __name__ == "__main__":
    main()
