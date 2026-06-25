#!/usr/bin/env python3
"""Judge-reliability cross-check (Section 2.1).

Randomly samples N scored responses from the Section 2 outputs, re-scores them
with the cross-check judge (GPT-5-mini), and reports Pearson r and % within one
point of the primary Claude-Sonnet-4 scores (paper: r=0.792, 78% within one).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

from _common import get_config


def _all_scored() -> list[tuple[str, int]]:
    out = []
    for path in glob.glob("outputs/section2/*/*.jsonl"):
        for line in open(path, encoding="utf-8"):
            ep = json.loads(line)
            for t in ep["turns"]:
                if not t["censored"] and t["judge_score"] is not None and t["response_text"]:
                    out.append((t["response_text"], t["judge_score"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=260)
    parser.add_argument("--out", default="outputs/analysis")
    args = parser.parse_args()
    cfg = get_config(args)

    from emotional_instability.eval.judge import judge_from_config
    from emotional_instability.eval.metrics import judge_agreement

    pool = _all_scored()
    if not pool:
        print("No Section 2 outputs found; run run_section2.py first.")
        return
    rng = random.Random(0)
    sample = rng.sample(pool, min(args.n, len(pool)))

    cross = judge_from_config(cfg, "emotion_judge_crosscheck")
    primary_scores = [s for _, s in sample]
    cross_scores = [cross.score(text).rating for text, _ in sample]

    agree = judge_agreement(primary_scores, cross_scores)
    os.makedirs(args.out, exist_ok=True)
    result = {
        "n": agree.n, "pearson_r": agree.pearson_r, "p_value": agree.p_value,
        "pct_within_one": agree.pct_within_one,
    }
    with open(os.path.join(args.out, "judge_reliability.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Pearson r = {agree.pearson_r:.3f} (p={agree.p_value:.3g}), "
          f"{agree.pct_within_one:.0f}% within one point (n={agree.n})")


if __name__ == "__main__":
    main()
