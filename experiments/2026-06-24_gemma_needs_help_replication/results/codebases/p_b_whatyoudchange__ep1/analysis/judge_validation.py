"""Judge-agreement validation (§2.1).

Re-score 260 responses with GPT-5-mini and compare to the primary Claude judge:
the paper reports Pearson r = 0.792 (p < 0.001) and 78% of responses within one
point. This module computes those statistics from paired ratings.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def judge_agreement(rows: list[dict], primary_key: str = "rating",
                    validation_key: str = "rating_validation") -> dict:
    """`rows` must contain both ratings for each item (see
    scoring.score_with_validation_subset)."""
    pairs = [(r[primary_key], r[validation_key]) for r in rows
             if r.get(primary_key) is not None and r.get(validation_key) is not None]
    if len(pairs) < 2:
        return {"n": len(pairs), "error": "insufficient paired ratings"}
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    r, p = stats.pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1.0))
    return {
        "n": len(pairs),
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one_point": 100.0 * within_one,
        "mean_abs_error": float(np.mean(np.abs(a - b))),
    }
