"""Judge reliability check (Section 2.1).

Randomly samples scored responses, re-scores them with the secondary judge, and
reports Pearson r and within-one-point agreement against the primary judge.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..config import SECONDARY_JUDGE_OPENROUTER_ID
from ..judge import SecondaryJudge
from .metrics import judge_agreement, load_scored


def run_reliability_check(
    result_path: str | Path,
    n: int = 260,
    seed: int = 0,
    secondary_id: str = SECONDARY_JUDGE_OPENROUTER_ID,
) -> dict:
    rows = load_scored(result_path)
    # Use the final response of each rollout and its primary final score.
    pool = [(r["turns"][-1]["assistant"], r["final_score"])
            for r in rows if r.get("final_score", -1) >= 0]
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))

    secondary = SecondaryJudge(secondary_id)
    primary_scores, secondary_scores = [], []
    for text, primary_score in sample:
        v = secondary.score(text)
        primary_scores.append(primary_score)
        secondary_scores.append(v.rating)

    result = judge_agreement(primary_scores, secondary_scores)
    result["secondary_judge"] = secondary_id
    return result
