#!/usr/bin/env python3
"""§2.1 reliability check: re-score a random subset of responses with the
secondary judge (GPT-5-mini via OpenRouter) and report Pearson r + % within one
point of the primary Claude-Sonnet ratings (paper: r=0.792, 78% within one)."""

from __future__ import annotations

import argparse
import json
import random

from emotional_instability.analysis.aggregate import judge_agreement, load_all_scores
from emotional_instability.config import load_config
from emotional_instability.judge import OpenRouterJudge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    n = int(cfg["judge"]["secondary_n"])
    df = load_all_scores(cfg, args.models)
    df = df[df["score"].notna()]

    rng = random.Random(cfg.seed)
    idx = rng.sample(range(len(df)), min(n, len(df)))
    subset = df.iloc[idx]

    secondary = OpenRouterJudge(cfg)
    primary_scores, secondary_scores = [], []
    for _, row in subset.iterrows():
        primary_scores.append(int(row["score"]))
        secondary_scores.append(secondary.score(row["response_text"]).rating)

    stats = judge_agreement(primary_scores, secondary_scores)
    print(json.dumps(stats, indent=2))
    with open(cfg.path_for("scores") / "judge_agreement.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
