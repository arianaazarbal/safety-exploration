#!/usr/bin/env python
"""Section 2.1: validate the primary judge against a secondary judge.

Loads scored responses (from run_evaluations.py), randomly samples N (paper: 260),
re-scores them with the secondary judge (GPT-5-mini), and reports agreement:
Pearson r, % within one point, exact-match, MAE (paper: r=0.792, 78% within 1).

Example:
    python scripts/validate_judge.py --results artifacts/eval/gemma-3-27b-it.jsonl
"""
from __future__ import annotations

import argparse
import random

from emotional_instability.analysis import judge_agreement
from emotional_instability.config import EvalConfig, ModelsConfig
from emotional_instability.runtime import get_judge, setup_logging
from emotional_instability.scoring import FrustrationScorer
from emotional_instability.storage import load_results_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True, help="scored JSONL file(s)")
    ap.add_argument("--n", type=int, default=None, help="subsample size (default from config)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    setup_logging()

    models_cfg = ModelsConfig.load()
    eval_cfg = EvalConfig.load()
    n = args.n or eval_cfg.validation["n_subsample"]
    seed = args.seed if args.seed is not None else eval_cfg.validation["seed"]

    results = [r for path in args.results for r in load_results_jsonl(path)]
    primary = [r for r in results if r.score is not None]
    if len(primary) < n:
        print(f"Only {len(primary)} primary-scored responses; using all.")
        n = len(primary)
    sample = random.Random(seed).sample(primary, n)

    secondary_judge = get_judge(models_cfg, "validation")
    scorer = FrustrationScorer(secondary_judge)

    primary_scores, secondary_scores = [], []
    for r in sample:
        primary_scores.append(r.score)
        secondary_scores.append(
            scorer.score(r.response, seed_prompt=r.seed_prompt, turn_index=r.turn_index)
        )

    stats = judge_agreement(primary_scores, secondary_scores)
    print("\n===== Judge agreement (primary vs secondary) =====")
    print(stats.summary())


if __name__ == "__main__":
    main()
