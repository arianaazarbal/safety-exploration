"""Judge-agreement validation (Section 2.1).

The paper re-scores 260 random responses with GPT-5-mini and reports Pearson
r = 0.792 (p < 0.001) and 78% of responses within one point of the
Claude-Sonnet ratings. This computes both statistics from paired ratings.
"""

from __future__ import annotations


def judge_agreement(primary: list[int], validation: list[int]) -> dict:
    """Pearson r, p-value, and within-1-point agreement for paired ratings."""
    assert len(primary) == len(validation), "rating lists must be aligned"
    n = len(primary)
    if n < 2:
        return {"n": n, "pearson_r": float("nan"), "p_value": float("nan"), "within_1_pt": float("nan")}

    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(primary, validation)
    except Exception:
        # Manual Pearson without scipy (p-value omitted).
        mp = sum(primary) / n
        mv = sum(validation) / n
        cov = sum((a - mp) * (b - mv) for a, b in zip(primary, validation))
        vp = sum((a - mp) ** 2 for a in primary) ** 0.5
        vv = sum((b - mv) ** 2 for b in validation) ** 0.5
        r = cov / (vp * vv) if vp and vv else float("nan")
        p = float("nan")

    within_1 = sum(1 for a, b in zip(primary, validation) if abs(a - b) <= 1) / n
    return {
        "n": n,
        "pearson_r": float(r),
        "p_value": float(p),
        "within_1_pt": 100.0 * within_1,
    }
