#!/usr/bin/env python
"""Section 2.1 validation: re-score a random 260-response sample with GPT-5-mini
and report Pearson r + within-1-point agreement vs the Claude-Sonnet-4 judge.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR
from src import analyze
from src.judge import make_check_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rolls = analyze.load_rollouts()
    turns = [t for r in rolls for t in r.turns if t.score is not None]
    random.Random(args.seed).shuffle(turns)
    sample = turns[: args.n]
    if not sample:
        print("No scored responses found — run Section 2 first.")
        return

    check = make_check_judge()
    primary, secondary = [], []
    for t in sample:
        primary.append(t.score)
        secondary.append(check.score(t.assistant).rating)

    stats = analyze.judge_agreement(primary, secondary)
    (RESULTS_DIR / "judge_agreement.json").write_text(json.dumps(stats, indent=2))
    print(f"Pearson r = {stats['pearson_r']:.3f} (p={stats['p_value']:.2e}), "
          f"within-1-point = {stats['within_1_point']*100:.1f}% (n={stats['n']})")


if __name__ == "__main__":
    main()
