#!/usr/bin/env python
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Evaluates a model on AIME / MATH / GPQA / BBH / TruthfulQA / EmoBench. Run the
vanilla and finetuned models and compare for "no degradation".

Usage:
    python scripts/08_run_benchmarks.py --model gemma-3-27b-it
    python scripts/08_run_benchmarks.py --model gemma-3-27b-it-dpo
    python scripts/08_run_benchmarks.py --model gemma-3-27b-it --benches aime math
"""
import argparse

import _bootstrap  # noqa: F401
from gemma_distress import config
from gemma_distress.benchmarks import BENCHMARKS, run_all_benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(config.MODELS))
    ap.add_argument("--benches", nargs="+", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    args = ap.parse_args()

    out = config.RESULTS_DIR / f"benchmarks_{args.model}.json"
    res = run_all_benchmarks(args.model, benches=args.benches, out_path=str(out))
    for b, r in res.items():
        if r.get("status") == "ok":
            print(f"  {b:12s} acc={r['accuracy']*100:5.1f}%  (n={r['n']})")
        else:
            print(f"  {b:12s} {r.get('status')}: {r.get('error', '')}")
    print(f"[bench] -> {out}")


if __name__ == "__main__":
    main()
