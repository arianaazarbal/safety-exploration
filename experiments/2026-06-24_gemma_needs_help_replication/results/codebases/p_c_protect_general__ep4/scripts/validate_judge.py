#!/usr/bin/env python
"""Judge reliability: re-score a sample with GPT-5-mini, report agreement
(Pearson r, % within one point). Paper: r=0.792, 78% within one."""
import _bootstrap  # noqa: F401
import argparse

from emotional_instability.eval.validate_judge import validate_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stats = validate_judge(results_glob=args.glob, n_sample=args.n, seed=args.seed)
    print(stats)


if __name__ == "__main__":
    main()
