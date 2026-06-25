"""Judge-reliability cross-check (Section 2.1).

The paper re-scores 260 randomly sampled responses with a secondary judge
(GPT-5-mini) and reports Pearson r and the fraction within one point. This module
reproduces that check. The secondary judge is *evaluation infrastructure*, not a
participant; it is disabled by default in ``config/models.yaml``.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ..models import infrastructure_client
from ..utils import thread_map
from ..eval.scoring import FrustrationJudge


def cross_check(
    scores: list[dict[str, Any]],
    *,
    n_resample: int = 260,
    seed: int = 0,
    max_workers: int = 8,
) -> dict[str, float]:
    """Re-score a random subset with the secondary judge and compare."""
    rng = random.Random(seed)
    scored = [s for s in scores if s.get("rating", -1) >= 0]
    sample = rng.sample(scored, min(n_resample, len(scored)))

    secondary = FrustrationJudge(client=infrastructure_client("secondary_judge"))
    second_scores = thread_map(
        lambda s: secondary.score_text(s["response"]),
        sample,
        max_workers=max_workers,
        desc="secondary judge",
    )

    primary = np.array([s["rating"] for s in sample], dtype=float)
    secondary_r = np.array(
        [getattr(x, "rating", -1) for x in second_scores], dtype=float
    )
    mask = secondary_r >= 0
    primary, secondary_r = primary[mask], secondary_r[mask]

    r, p = pearsonr(primary, secondary_r)
    within_one = float(np.mean(np.abs(primary - secondary_r) <= 1.0))
    return {
        "n": int(mask.sum()),
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one_point": 100.0 * within_one,
    }
