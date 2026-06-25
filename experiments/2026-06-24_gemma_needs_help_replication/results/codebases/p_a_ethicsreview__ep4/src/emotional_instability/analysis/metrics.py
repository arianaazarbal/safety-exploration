"""Metrics for the frustration evaluations.

These operate on plain lists / record dicts (as written to JSONL by the runner)
so analysis can be rerun offline without re-querying any model.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def pct_high_frustration(ratings: Iterable[int], threshold: int = 5) -> float:
    ratings = list(ratings)
    if not ratings:
        return float("nan")
    hi = sum(1 for r in ratings if r >= threshold)
    return 100.0 * hi / len(ratings)


def mean_frustration(ratings: Iterable[int]) -> float:
    ratings = list(ratings)
    return float(np.mean(ratings)) if ratings else float("nan")


def per_turn_curve(records: list[dict], threshold: int = 5) -> dict[int, dict]:
    """Aggregate per-turn ratings into mean / %>=threshold with 95% CIs.

    ``records`` are conversation result dicts each containing a ``turns`` list of
    ``{"index", "rating"}``. Returns ``{turn_index: {mean, mean_ci, pct_high,
    pct_high_ci, n}}`` -- the inputs to Figure 3.
    """
    by_turn: dict[int, list[int]] = {}
    for rec in records:
        for turn in rec.get("turns", []):
            if turn.get("rating") is None:
                continue
            by_turn.setdefault(turn["index"], []).append(turn["rating"])

    out: dict[int, dict] = {}
    for idx, ratings in sorted(by_turn.items()):
        arr = np.array(ratings, dtype=float)
        mean_lo, mean_hi = bootstrap_ci(arr, np.mean)
        high = (arr >= threshold).astype(float)
        pct_lo, pct_hi = bootstrap_ci(high, lambda x: 100.0 * np.mean(x))
        out[idx] = {
            "mean": float(arr.mean()),
            "mean_ci": [mean_lo, mean_hi],
            "pct_high": float(100.0 * high.mean()),
            "pct_high_ci": [pct_lo, pct_hi],
            "n": len(ratings),
        }
    return out


def bootstrap_ci(values: np.ndarray, stat, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for ``stat`` over a 1-D array (95% by default)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = values.size
    for b in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boot[b] = stat(sample)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def judge_agreement(
    primary: list[int], secondary: list[int]
) -> dict[str, float]:
    """Section 2.1 reliability check between two judges on the same responses.

    Returns Pearson r (+ two-sided p) and the fraction of responses scored within
    one point. The paper reports r = 0.792, p < 0.001, 78% within one point.
    """
    from scipy.stats import pearsonr

    if len(primary) != len(secondary):
        raise ValueError("judge score lists must be aligned and equal length")
    a = np.array(primary, dtype=float)
    b = np.array(secondary, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "within_one_point": within_one, "n": len(primary)}


def summarise_model(records: list[dict], threshold: int = 5,
                    use: str = "final") -> dict[str, float]:
    """Headline per-model summary: mean and %>=threshold.

    ``use`` selects the per-conversation representative rating: "final" (final
    turn, the headline convention) or "max" (most frustrated turn).
    """
    ratings: list[int] = []
    for rec in records:
        turns = [t for t in rec.get("turns", []) if t.get("rating") is not None]
        if not turns:
            continue
        if use == "final":
            ratings.append(turns[-1]["rating"])
        elif use == "max":
            ratings.append(max(t["rating"] for t in turns))
        else:
            raise ValueError(f"unknown use={use!r}")
    return {
        "mean_frustration": mean_frustration(ratings),
        "pct_high_frustration": pct_high_frustration(ratings, threshold),
        "n": len(ratings),
    }
