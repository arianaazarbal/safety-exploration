#!/usr/bin/env python
"""Section 4 evaluation: Petri elicitation, capability benchmarks, and re-running
the Section-2 eval on finetuned adapters.

    # frustration eval of the DPO model (reuses Section 2 harness)
    python scripts/run_section4_eval.py eval --model gemma-3-27b-it \
        --adapter outputs/adapters/dpo

    # Petri open-ended elicitation across models
    python scripts/run_section4_eval.py petri --models gemma-3-27b-it gemini-2.5-flash

    # capability benchmarks (vanilla vs DPO)
    python scripts/run_section4_eval.py benchmarks --model gemma-3-27b-it \
        --adapter outputs/adapters/dpo --n 100
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--adapter", default=None)
    e.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("petri")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--adapter", default=None,
                   help="adapter applied to the FIRST model in --models")

    b = sub.add_parser("benchmarks")
    b.add_argument("--model", required=True)
    b.add_argument("--adapter", default=None)
    b.add_argument("--n", type=int, default=100)
    b.add_argument("--benchmarks", nargs="+", default=None)

    args = ap.parse_args()

    if args.cmd == "eval":
        from gemma_distress.eval.run_eval import print_summary, run_evaluation
        path = run_evaluation(args.model, adapter_path=args.adapter,
                              limit=args.limit)
        print_summary(path)
    elif args.cmd == "petri":
        from gemma_distress.petri.run_petri import run_petri, summarize_petri
        adapters = {args.models[0]: args.adapter} if args.adapter else None
        path = run_petri(args.models, adapter_paths=adapters)
        print("\n=== Petri summary (mean transcript score per emotion) ===")
        for model, emo in summarize_petri(path).items():
            for e, v in emo.items():
                print(f"{model:24s} {e:12s} mean={v['mean']:.2f} "
                      f"CI={v['ci']} n={v['n']}")
    elif args.cmd == "benchmarks":
        from gemma_distress.benchmarks.run_benchmarks import run_all_benchmarks
        run_all_benchmarks(args.model, adapter_path=args.adapter, n=args.n,
                           benchmarks=args.benchmarks)


if __name__ == "__main__":
    main()
