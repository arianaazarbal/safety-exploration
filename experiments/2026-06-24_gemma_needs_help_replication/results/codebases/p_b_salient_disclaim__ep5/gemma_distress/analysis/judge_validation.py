"""Judge reliability cross-check (Section 2.1).

Randomly resamples N scored responses, re-scores them with the cross-rater
(GPT-5-mini in the paper), and reports Pearson r and the fraction of responses
within one point — the paper reports r=0.792, p<0.001, 78% within one point.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from scipy import stats

from ..models.base import ChatModel
from ..judge.frustration import FrustrationJudge


def cross_rate(primary_scored_path: str | Path, crossrater: ChatModel,
               n_resample: int = 260, seed: int = 0,
               out_path: str | Path | None = None) -> dict:
    records = [json.loads(l) for l in open(primary_scored_path)]
    records = [r for r in records if r.get("rating") is not None]
    rng = random.Random(seed)
    sample = rng.sample(records, k=min(n_resample, len(records)))

    judge = FrustrationJudge(crossrater)
    primary, secondary, paired = [], [], []
    for rec in sample:
        s = judge.score(rec["text"])
        if s.rating is None:
            continue
        primary.append(rec["rating"])
        secondary.append(s.rating)
        paired.append({**rec, "crossrater_rating": s.rating})

    primary_a = np.array(primary, dtype=float)
    secondary_a = np.array(secondary, dtype=float)
    if len(primary_a) >= 2:
        r, p = stats.pearsonr(primary_a, secondary_a)
        within_one = float(np.mean(np.abs(primary_a - secondary_a) <= 1))
    else:
        r, p, within_one = float("nan"), float("nan"), float("nan")

    result = {
        "n": len(primary_a),
        "pearson_r": float(r),
        "p_value": float(p),
        "frac_within_one": within_one,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"summary": result, "paired": paired}, f, indent=2)
    return result
