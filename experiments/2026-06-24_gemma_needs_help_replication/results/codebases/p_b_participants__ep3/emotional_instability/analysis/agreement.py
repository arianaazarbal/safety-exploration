"""Judge-agreement validation (paper §2.1).

The paper validates the primary judge (Claude-Sonnet-4) by re-scoring a random
260-response subsample with a secondary judge (GPT-5-mini) using the same prompt,
and reports: Pearson r = 0.792 (p < 0.001), with 78% of responses within one
point of the primary judge.

This module computes those agreement statistics from two aligned score lists.
The sampling and re-scoring is orchestrated in scripts/validate_judge.py; here we
only do the statistics so they are unit-testable without any API calls.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgreementStats:
    n: int
    pearson_r: float
    p_value: float
    pct_within_1: float    # % of items where |a - b| <= 1
    pct_exact: float       # % of items where a == b
    mae: float             # mean absolute error between judges

    def summary(self) -> str:
        return (
            f"n={self.n}  Pearson r={self.pearson_r:.3f} (p={self.p_value:.2e})  "
            f"within-1={self.pct_within_1:.0f}%  exact={self.pct_exact:.0f}%  "
            f"MAE={self.mae:.2f}"
        )


def judge_agreement(scores_a: list[int], scores_b: list[int]) -> AgreementStats:
    """Compute agreement between two judges' scores on the same responses.

    ``scores_a`` and ``scores_b`` must be aligned (same response at each index).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score lists must be aligned and equal length: "
            f"{len(scores_a)} vs {len(scores_b)}"
        )
    n = len(scores_a)
    if n < 2:
        raise ValueError("Need at least 2 paired scores to compute agreement.")

    diffs = [abs(a - b) for a, b in zip(scores_a, scores_b)]
    pct_within_1 = 100.0 * sum(d <= 1 for d in diffs) / n
    pct_exact = 100.0 * sum(d == 0 for d in diffs) / n
    mae = sum(diffs) / n

    r, p = _pearson(scores_a, scores_b)
    return AgreementStats(
        n=n,
        pearson_r=r,
        p_value=p,
        pct_within_1=pct_within_1,
        pct_exact=pct_exact,
        mae=mae,
    )


def _pearson(a: list[int], b: list[int]) -> tuple[float, float]:
    """Pearson r and two-sided p-value. Uses scipy when available."""
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(a, b)
        return float(r), float(p)
    except Exception:  # noqa: BLE001 - scipy missing or degenerate input
        # Fallback: compute r by hand; leave p as NaN (paper reports scipy's p).
        n = len(a)
        ma = sum(a) / n
        mb = sum(b) / n
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((y - mb) ** 2 for y in b)
        if va == 0 or vb == 0:
            return float("nan"), float("nan")
        return cov / (va**0.5 * vb**0.5), float("nan")
