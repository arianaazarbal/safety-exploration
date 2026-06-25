"""Inter-judge agreement (paper Section 2.1).

Reproduces the validation statistics: Pearson r between the primary
(Claude-Sonnet-4) and secondary (GPT-5-mini) judges over a sample of responses,
and the fraction of responses scored within one point. The paper reports
r = 0.792 (p < 0.001) and 78% within one point on a 260-response sample.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JudgeAgreement:
    n: int
    pearson_r: float
    p_value: float
    within_one_point: float        # fraction in [0, 1]
    mean_abs_diff: float


def compute_agreement(primary: list[int], secondary: list[int]) -> JudgeAgreement:
    """Compute agreement stats between two equal-length lists of integer scores."""
    if len(primary) != len(secondary):
        raise ValueError("score lists must be the same length")
    if not primary:
        raise ValueError("no scores to compare")

    import numpy as np
    from scipy import stats

    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)

    # scipy.pearsonr is undefined if either series is constant; guard it.
    if a.std() == 0 or b.std() == 0:
        r, p = float("nan"), float("nan")
    else:
        r, p = stats.pearsonr(a, b)

    diffs = np.abs(a - b)
    return JudgeAgreement(
        n=len(primary),
        pearson_r=float(r),
        p_value=float(p),
        within_one_point=float((diffs <= 1).mean()),
        mean_abs_diff=float(diffs.mean()),
    )
