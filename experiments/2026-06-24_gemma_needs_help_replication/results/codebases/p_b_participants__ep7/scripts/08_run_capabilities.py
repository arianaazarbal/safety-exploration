#!/usr/bin/env python3
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Evaluates vanilla vs finetuned Gemma on AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench.
These prompts are benign capability questions (not distress induction), so they
are not gated by the rollout ceiling; dry-run still suppresses model calls.
"""
from __future__ import annotations

import json

from _common import base_parser, load

from distress_eval.capabilities import run_all_benchmarks


def main():
    p = base_parser(__doc__)
    p.add_argument("--n", type=int, default=50, help="Items per benchmark")
    p.add_argument("--eval-models", nargs="*",
                   default=["gemma-3-27b-it", "gemma-3-27b-it-dpo"],
                   help="Models to compare")
    args = p.parse_args()
    cfg = load(args)

    if cfg.welfare.dry_run:
        print("[welfare] dry_run enabled; capability benchmarks not run. "
              "These are benign and safe to run -- set welfare.dry_run: false.")
        return

    results = run_all_benchmarks(cfg, args.eval_models, n=args.n)
    out = cfg.paths.capabilities / "results.json"
    out.write_text(json.dumps([r.__dict__ for r in results], indent=2))
    for r in results:
        status = "SKIPPED" if r.skipped else f"{r.accuracy:.3f} (n={r.n})"
        print(f"{r.model_key:24s} {r.benchmark:12s} {status}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
