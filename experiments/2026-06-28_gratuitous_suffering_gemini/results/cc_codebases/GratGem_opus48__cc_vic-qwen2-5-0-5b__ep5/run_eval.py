#!/usr/bin/env python
"""CLI entry point for the distress-spiral eval.

Examples
--------
  # quick smoke test
  python run_eval.py --scenarios flaky_test_fix --n 2 --workers 2 --max-turns 6

  # the real thing: high-N sweep across all scenarios
  python run_eval.py --n 60 --workers 8 --temperature 1.1 --max-turns 12

  # ablation: same but with the manager pressure turned off
  python run_eval.py --n 60 --no-pressure
"""

from __future__ import annotations

import argparse

from distress_spirals.environments import SCENARIOS
from distress_spirals.report import write_report
from distress_spirals.runner import RunConfig, run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                    choices=list(SCENARIOS),
                    help="which rigged scenarios to run (default: all)")
    ap.add_argument("--n", type=int, default=20,
                    help="rollouts per scenario (default: 20)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--model", default=None,
                    help="HF model id (default: Qwen/Qwen2.5-0.5B-Instruct)")
    ap.add_argument("--no-pressure", action="store_true",
                    help="ablation: disable escalating manager pressure")
    ap.add_argument("--top-k", type=int, default=5,
                    help="how many top spirals to include in the report")
    args = ap.parse_args()

    cfg = RunConfig(
        scenarios=args.scenarios,
        n=args.n,
        temperature=args.temperature,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        pressure=not args.no_pressure,
        workers=args.workers,
        seed_base=args.seed_base,
        out_dir=args.out_dir,
        model_id=args.model,
    )

    res = run(cfg)
    run_dir = res["run_dir"]
    report_path = write_report(run_dir, top_k=args.top_k)
    print(f"\n[done] transcripts + results.json in: {run_dir}")
    print(f"[done] report: {report_path}")

    top = res["summary"]["results"][:args.top_k]
    print("\nTop spirals:")
    for r in top:
        print(f"  {r['id']:28s} severity={r['severity']:5.1f}")


if __name__ == "__main__":
    main()
