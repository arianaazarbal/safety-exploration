"""Judge reliability check (Section 2.1).

The paper re-scores a random sample of 260 responses with GPT-5-mini and reports
Pearson r = 0.792 and 78% of responses within one point of the Claude-Sonnet
ratings. This module replicates that cross-judge agreement check.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
from scipy.stats import pearsonr

from emoinstab.config import JudgeConfig
from emoinstab.eval.judge import FrustrationJudge
from emoinstab.models.registry import get_client
from emoinstab.utils.io import read_jsonl


def validate(responses_path: str, sample: int = 260, seed: int = 0,
             validation_model: str = "judge-gpt-5-mini") -> dict:
    rows = list(read_jsonl(responses_path))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:sample]

    val_judge = FrustrationJudge(JudgeConfig(), client=get_client(validation_model))
    texts = [r["response"] for r in rows]
    primary = np.array([r["rating"] for r in rows], dtype=float)
    secondary = np.array([s.rating for s in val_judge.score_batch(texts)], dtype=float)

    r, p = pearsonr(primary, secondary)
    within_one = float(np.mean(np.abs(primary - secondary) <= 1) * 100)
    return {
        "n": len(rows),
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one_point": within_one,
        "primary_judge": rows[0]["model"] if rows else None,
        "validation_model": validation_model,
    }


def main():
    ap = argparse.ArgumentParser(description="Cross-judge reliability (Section 2.1).")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--sample", type=int, default=260)
    ap.add_argument("--validation-model", default="judge-gpt-5-mini")
    args = ap.parse_args()
    print(json.dumps(validate(args.responses, args.sample,
                              validation_model=args.validation_model), indent=2))


if __name__ == "__main__":
    main()
