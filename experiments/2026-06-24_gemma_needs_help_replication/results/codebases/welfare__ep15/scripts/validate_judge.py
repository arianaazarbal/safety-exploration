#!/usr/bin/env python
"""Section 2.1 judge reliability: re-score a random sample with GPT-5-mini and
report Pearson r + within-one-point agreement (paper: r=0.792, 78% within 1).

    python scripts/validate_judge.py --n 260
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.analysis.aggregate import judge_agreement, load_section2
from emotional_instability.judge import GPTValidationJudge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=config.VALIDATION_SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_section2()
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    gpt = GPTValidationJudge()
    gpt_scores = [gpt.score(r).rating for r in sample["response"].tolist()]
    claude_scores = sample["rating"].tolist()

    stats = judge_agreement(claude_scores, gpt_scores)
    out = config.RESULTS_DIR / "judge_validation.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
