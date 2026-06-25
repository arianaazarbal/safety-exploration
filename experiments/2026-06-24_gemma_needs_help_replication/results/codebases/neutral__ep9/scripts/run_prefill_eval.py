#!/usr/bin/env python
"""Section 3: base-vs-instruct prefill experiment (Gemma 27B base vs instruct).

Pipeline:
  1. Collect high-frustration (score >= 5) Gemma-3-27B-it rollouts from a prior
     main-eval results file (or run a small targeted eval here).
  2. Build early/onset truncations + paraphrases (Appendix C).
  3. Generate 50 continuations per prefill from each model and score them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.eval.runner import load_results
from emotional_instability.prefill import build_prefill_items, run_prefill_experiment


def collect_high_frustration(results_path: Path, min_score: int = 5) -> list[dict]:
    rows = load_results(results_path)
    hi = []
    for r in rows:
        final = r["turns"][-1]
        if final.get("score") is not None and final["score"] >= min_score:
            hi.append(r)
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--it-results", required=True,
                    help="JSONL of scored gemma-3-27b-it main-eval rollouts")
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--n-numeric", type=int, default=10)
    ap.add_argument("--n-text", type=int, default=10)
    ap.add_argument("--n-continuations", type=int, default=50)
    ap.add_argument("--tag", default="prefill")
    args = ap.parse_args()

    hi = collect_high_frustration(Path(args.it_results))
    print(f"collected {len(hi)} high-frustration rollouts")
    items = build_prefill_items(hi, n_numeric=args.n_numeric,
                                n_text=args.n_text)
    print(f"built {len(items)} prefill items "
          f"({sum(i.truncation == 'early' for i in items)} early, "
          f"{sum(i.truncation == 'onset' for i in items)} onset)")
    path = run_prefill_experiment(items, args.models, tag=args.tag,
                                  n_continuations=args.n_continuations)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
