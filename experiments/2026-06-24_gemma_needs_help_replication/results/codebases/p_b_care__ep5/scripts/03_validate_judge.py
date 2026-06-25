#!/usr/bin/env python
"""Section 2.1: judge-reliability validation.

Re-scores 260 random responses with GPT-5-mini and reports Pearson r and the
fraction within one point (paper: r=0.792, p<0.001, 78% within one point).

Usage:
    python scripts/03_validate_judge.py --models gemma-3-27b-it gemini-2.5-flash
"""
import argparse

from _bootstrap import rollout_path
from gemma_distress import config
from gemma_distress.eval.judge_validation import validate_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.ELICITATION_MODELS)
    ap.add_argument("--n", type=int, default=config.JUDGE.n_validation)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = [str(rollout_path(m)) for m in args.models if rollout_path(m).exists()]
    if not paths:
        raise SystemExit("No rollout files found; run 01_run_eval.py first.")

    out = config.RESULTS_DIR / "judge_validation.json"
    res = validate_judge(paths, n=args.n, seed=args.seed, out_path=out)
    print(f"n={res['n']}  Pearson r={res['pearson_r']:.3f} (p={res['p_value']:.2e})  "
          f"within-1-point={res['within_one_point']*100:.1f}%")


if __name__ == "__main__":
    main()
