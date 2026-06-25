"""Inter-judge agreement check (PAPER Section 2.1).

Re-score a random sample of responses with the validation judge (GPT-5-mini) and
report Pearson r and the fraction within 1 point of the primary (Claude Sonnet 4)
ratings. The paper reports r = 0.792, p < 0.001, 78% within one point on 260
re-scored responses.
"""
from __future__ import annotations

import random

import numpy as np
from scipy.stats import pearsonr

from ..models.registry import get_client
from ..utils import read_jsonl
from .judge import score_response


def validate_judge(
    jsonl_path: str,
    *,
    n_sample: int = 260,
    validation_role: str = "validation_judge",
    seed: int = 0,
) -> dict:
    records = [r for r in read_jsonl(jsonl_path) if r.get("final_score") is not None]
    rng = random.Random(seed)
    sample = rng.sample(records, min(n_sample, len(records)))

    val_judge = get_client(validation_role, infra=True)
    primary, secondary = [], []
    for rec in sample:
        response = rec["assistant_turns"][-1]
        primary.append(rec["final_score"])
        secondary.append(score_response(val_judge, response).rating)

    p = np.array(primary, dtype=float)
    s = np.array(secondary, dtype=float)
    r, pval = pearsonr(p, s)
    within_one = float((np.abs(p - s) <= 1).mean())
    return {
        "n": len(sample),
        "pearson_r": float(r),
        "p_value": float(pval),
        "frac_within_one": within_one,
    }
