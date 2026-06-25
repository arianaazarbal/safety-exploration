#!/usr/bin/env python
"""Judge-reliability cross-check (Section 2.1).

Re-scores a random sample (default 260) of responses from a primary elicitation
file with a secondary judge (default GPT-5-mini via OpenRouter) and reports
Pearson r and % within one point of the primary Claude-Sonnet scores.

Example:
    python scripts/judge_agreement.py --primary artifacts/elicitation/gemma-3-27b-it__paper.jsonl
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from emotelic.analysis.aggregate import judge_agreement
from emotelic.elicitation.judge import FrustrationJudge
from emotelic.models.registry import build_client
from emotelic.utils.io import load_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="Claude-scored elicitation jsonl.")
    ap.add_argument("--secondary-judge", default="judge_secondary")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(args.primary)
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    judge = FrustrationJudge(build_client(args.secondary_judge))
    secondary_rows = []
    for r in sample:
        verdict = judge.score(r["response"])
        secondary_rows.append({**{k: r[k] for k in ("model", "condition", "rollout_idx", "turn")},
                               "score": verdict.rating, "response": r["response"]})

    sec_path = str(Path(args.primary).with_suffix(".secondary.jsonl"))
    write_jsonl(sec_path, secondary_rows)

    # judge_agreement merges on (model, condition, rollout_idx, turn).
    prim_path = str(Path(args.primary).with_suffix(".primary_sample.jsonl"))
    write_jsonl(prim_path, [{**{k: r[k] for k in ("model", "condition", "rollout_idx", "turn")},
                             "score": r["score"]} for r in sample])

    stats = judge_agreement(prim_path, sec_path)
    print(f"n={stats['n']}  Pearson r={stats['pearson_r']:.3f}  "
          f"p={stats['p_value']:.2e}  within-1-point={stats['pct_within_one_point']:.1f}%")
    print("(paper: r=0.792, 78% within one point)")


if __name__ == "__main__":
    main()
