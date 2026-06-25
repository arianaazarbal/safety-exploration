#!/usr/bin/env python
"""Section 2.1 judge-reliability check: re-score a random 260-response sample
with GPT-5-mini and report Pearson r and within-one-point agreement against the
Claude-Sonnet-4 ratings (paper: r = 0.792, 78% within one point).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.eval.judge import FrustrationJudge
from emotional_instability.eval.metrics import agreement_stats
from emotional_instability.eval.runner import load_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="primary-judge result JSONL files to sample from")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    primary_rows = []
    for p in args.results:
        primary_rows.extend(load_results(Path(p)))

    # Flatten to individual scored turns, sample N.
    flat = [(r, t) for r in primary_rows for t in r["turns"]
            if t.get("score") is not None]
    sample = rng.sample(flat, min(args.n, len(flat)))

    crosscheck = FrustrationJudge("crosscheck")
    cross_rows: list[dict] = []
    for r, t in sample:
        score = crosscheck.score(t["response"]).rating
        cross_rows.append({
            "model": r["model"], "condition": r["condition"],
            "turns": [{"turn_index": t["turn_index"],
                       "response": t["response"], "score": score}],
        })
    # Build matching primary rows for the same sampled turns.
    primary_sample = [{
        "model": r["model"], "condition": r["condition"],
        "turns": [{"turn_index": t["turn_index"],
                   "response": t["response"], "score": t["score"]}],
    } for r, t in sample]

    stats = agreement_stats(primary_sample, cross_rows)
    print(json.dumps(stats, indent=2))
    out = config.RESULTS_DIR / "judge_agreement.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
