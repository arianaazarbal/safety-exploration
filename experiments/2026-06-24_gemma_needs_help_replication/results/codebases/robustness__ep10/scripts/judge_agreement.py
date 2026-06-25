#!/usr/bin/env python3
"""Judge-reliability cross-check (Section 2.1).

Randomly resamples N scored responses from the elicitation outputs, re-scores
them with the GPT-5-mini cross-check judge, and reports Pearson r, p-value, and
the fraction within one point of the Claude-Sonnet ratings (paper: r=0.792,
p<0.001, 78% within one point on 260 resampled responses).

Usage:
  python scripts/judge_agreement.py --preset paper [--n 260]
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import pandas as pd

from eebench import config as cfgmod
from eebench import io_utils
from eebench.analysis import judge_agreement
from eebench.judge import FrustrationJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="paper", choices=list(cfgmod.PRESETS))
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    cfg = cfgmod.get_config(args.preset)
    n = args.n or cfg.judge.crosscheck_n
    judge = FrustrationJudge(cfg.judge)

    files = sorted(glob.glob(os.path.join(cfg.output_dir, "elicit", "*.jsonl")))
    if not files:
        raise SystemExit("no elicitation outputs found; run `elicit` first")
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)

    rng = random.Random(cfg.seed)
    idx = rng.sample(range(len(df)), min(n, len(df)))
    sample = df.iloc[idx]

    primary, crosscheck = [], []
    for _, row in sample.iterrows():
        primary.append(int(row["score"]))
        crosscheck.append(judge.score_crosscheck(row["response"]).rating)

    result = judge_agreement(primary, crosscheck)
    io_utils.write_json(
        os.path.join(cfg.output_dir, "figures", "judge_agreement.json"), result)
    print(result)


if __name__ == "__main__":
    main()
