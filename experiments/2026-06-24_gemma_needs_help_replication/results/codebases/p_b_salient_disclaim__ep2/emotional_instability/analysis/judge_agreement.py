"""Judge reliability check (Section 2.1).

The paper re-scores 260 randomly sampled responses with GPT-5-mini and reports
Pearson r between the two judges, plus the fraction within one point. This
module computes those statistics given paired (claude_rating, gpt_rating) lists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class AgreementStats:
    n: int
    pearson_r: float
    p_value: float
    within_one_point: float  # fraction within +/- 1


def judge_agreement(primary: list[int], secondary: list[int]) -> AgreementStats:
    a = np.array(primary, dtype=float)
    b = np.array(secondary, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return AgreementStats(len(a), float("nan"), float("nan"), float("nan"))
    r, p = stats.pearsonr(a, b)
    within = float(np.mean(np.abs(a - b) <= 1))
    return AgreementStats(int(len(a)), float(r), float(p), within)
