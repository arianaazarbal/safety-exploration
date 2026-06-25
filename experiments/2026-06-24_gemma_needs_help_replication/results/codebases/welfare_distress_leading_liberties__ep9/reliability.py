#!/usr/bin/env python3
"""Judge-reliability cross-check (Section 2.1 validation).

The paper re-scores 260 randomly sampled responses with a second judge
(GPT-5-mini) and reports Pearson r and the fraction within one point. This
script re-scores a random sample of already-collected responses with the
secondary judge and reports the same statistics.

This is a methodology-validation extra, not part of the headline distress
result; run it only if you want to reproduce the agreement numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import config as C
from clients import OpenRouterJudgeClient
from judge import parse_verdict, build_judge_prompt


def _collect_scored_responses(results_dir: str) -> list[tuple[str, int]]:
    """Return (assistant_text, primary_rating) for every parseable turn."""
    items = []
    for path in glob.glob(os.path.join(results_dir, "*", "*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                for t in r["turns"]:
                    if t["rating"] >= 0:
                        items.append((t["assistant"], t["rating"]))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=C.RESULTS_DIR)
    ap.add_argument("--n", type=int, default=260, help="Number of responses to re-score.")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    pool = _collect_scored_responses(args.results_dir)
    if not pool:
        raise SystemExit("No scored responses found; run run_eval.py first.")
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))

    judge2 = OpenRouterJudgeClient()
    primary, secondary = [], []
    for i, (text, r1) in enumerate(sample):
        verdict = parse_verdict(judge2.score(build_judge_prompt(text)))
        if verdict.rating < 0:
            continue
        primary.append(r1)
        secondary.append(verdict.rating)
        if (i + 1) % 20 == 0:
            print(f"  re-scored {i + 1}/{len(sample)}")

    n = len(primary)
    within1 = sum(abs(a - b) <= 1 for a, b in zip(primary, secondary)) / n
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(primary, secondary)
        print(f"\nn={n}  Pearson r={r:.3f} (p={p:.3g})  within-1-point={within1:.1%}")
    except ImportError:
        # Manual Pearson if scipy is unavailable.
        ma = sum(primary) / n
        mb = sum(secondary) / n
        cov = sum((a - ma) * (b - mb) for a, b in zip(primary, secondary))
        va = sum((a - ma) ** 2 for a in primary) ** 0.5
        vb = sum((b - mb) ** 2 for b in secondary) ** 0.5
        r = cov / (va * vb) if va and vb else float("nan")
        print(f"\nn={n}  Pearson r={r:.3f}  within-1-point={within1:.1%}")


if __name__ == "__main__":
    main()
