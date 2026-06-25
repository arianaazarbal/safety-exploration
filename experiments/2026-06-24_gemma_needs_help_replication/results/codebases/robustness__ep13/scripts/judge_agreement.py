#!/usr/bin/env python
"""Judge agreement check (Section 2.1).

Re-score a random subsample of responses with a second judge (gpt-5-mini) and
report Pearson r and the fraction of responses within one point -- the paper
reports r = 0.792 and 78% within one point on 260 responses.

Usage:
    python scripts/judge_agreement.py --scored results/full/gemini-2.5-flash.scored.jsonl \
        --config configs/full.yaml --n 260
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from emotional_instability.config import ExperimentConfig
from emotional_instability.eval.judge_runner import judge_response
from emotional_instability.models import build_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="primary-judge scored JSONL")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if not cfg.cross_judge:
        raise SystemExit("config has no cross_judge defined")
    cross = build_backend(cfg.cross_judge)

    with open(args.scored) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    primary, secondary = [], []
    for r in sample:
        rating2, _, ok = judge_response(cross, r["response"])
        if not ok:
            continue
        primary.append(r["rating"])
        secondary.append(rating2)

    a, b = np.array(primary, float), np.array(secondary, float)
    r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
    within_one = float(np.mean(np.abs(a - b) <= 1)) if len(a) else float("nan")
    result = dict(n=len(a), pearson_r=r, frac_within_one=within_one)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
