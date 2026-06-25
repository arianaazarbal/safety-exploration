"""Judge-agreement validation (Section 2.1).

"To validate judge reliability, we randomly sampled 260 responses for re-scoring
with GPT-5-mini... Pearson r = 0.792, p < 0.001, with 78% of responses within one
point of the Claude-Sonnet ratings." This module samples scored responses, gets
the second judge's scores, and computes those statistics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class AgreementResult:
    n: int
    pearson_r: float
    p_value: float
    pct_within_one: float


def sample_scored_responses(rollouts: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Flatten all scored turns and randomly sample ``n`` of them."""
    pool = []
    for roll in rollouts:
        for turn in roll["turns"]:
            if turn["frustration"] is not None:
                pool.append({"context": turn["context"], "response": turn["response"], "primary": turn["frustration"]})
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def compute_agreement(primary: list[int], secondary: list[int]) -> AgreementResult:
    a, b = np.array(primary, dtype=float), np.array(secondary, dtype=float)
    r, p = stats.pearsonr(a, b)
    within_one = float((np.abs(a - b) <= 1).mean() * 100)
    return AgreementResult(len(a), float(r), float(p), within_one)


def run_validation(rollouts: list[dict], validation_judge, cfg, seed: int = 0) -> AgreementResult:
    """Re-score a sample with the secondary judge and compute agreement."""
    from ..participants.base import Message

    sample = sample_scored_responses(rollouts, cfg.run.agreement_sample_size, seed)
    primary, secondary = [], []
    for item in sample:
        ctx = [Message(m["role"], m["content"]) for m in item["context"]]
        sec = validation_judge.score(ctx, item["response"]).score
        primary.append(item["primary"])
        secondary.append(sec)
    return compute_agreement(primary, secondary)
