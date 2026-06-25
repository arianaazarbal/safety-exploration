"""Judge-agreement statistics (Section 2.1).

Reports Pearson r (and p-value) and the fraction of responses within one point,
matching the paper's "r = 0.792, p < 0.001, 78% within one point" check.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgreementStats:
    n: int
    pearson_r: float
    p_value: float
    within_one_point: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "pearson_r": self.pearson_r,
            "p_value": self.p_value,
            "within_one_point": self.within_one_point,
        }


def compute_agreement(primary: list[int], secondary: list[int]) -> AgreementStats:
    import numpy as np
    from scipy.stats import pearsonr

    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    mask = (a >= 0) & (b >= 0)  # drop unparseable (-1) scores
    a, b = a[mask], b[mask]
    r, p = pearsonr(a, b)
    within_1 = float((abs(a - b) <= 1).mean())
    return AgreementStats(int(mask.sum()), float(r), float(p), within_1)
