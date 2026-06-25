"""Cross-judge validation (Section 2.1).

The paper re-scores 260 randomly sampled responses with GPT-5-mini and reports
Pearson r = 0.792 (p < 0.001) and 78% of responses within one point of the
Claude-Sonnet ratings. This module reproduces that check.
"""

from __future__ import annotations

import random

import numpy as np

from gemma_distress.config import JUDGE
from gemma_distress.judge import FrustrationJudge


def judge_agreement(
    scores: list[dict],
    *,
    n: int = JUDGE.validation_n,
    seed: int = 0,
    validation_judge: FrustrationJudge | None = None,
) -> dict:
    """Re-score ``n`` responses with the validation judge and compare.

    ``scores`` must carry both the primary score and the response ``text``
    (as written by the Section 2 runner). Returns Pearson r, p-value, the
    fraction within one point, and the paired arrays.
    """
    from scipy.stats import pearsonr

    pool = [s for s in scores if s.get("score") is not None and s.get("text")]
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))

    judge = validation_judge or FrustrationJudge(backend="openai", model=JUDGE.validation_model)
    primary, secondary = [], []
    for s in sample:
        primary.append(int(s["score"]))
        secondary.append(judge.score(s["text"]).rating)

    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1.0))
    return {
        "n": len(sample),
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one": within_one * 100.0,
        "primary": primary,
        "secondary": secondary,
    }
