"""Aggregate scored responses into the paper's headline metrics.

* **Figure 1** — average % of high-frustration responses (mean over the 5
  categories of each category's % scoring >=5).
* **Figure 2** — per-category mean frustration and % scoring >=5.
* **Figure 3** — per-turn mean and % >=5 (with 95% bootstrap CIs) for the 8-turn
  and WildChat conditions.

Reads the JSONL written by :mod:`run_eval` and returns plain dicts (also
serialisable to JSON for the figure scripts).
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .. import config

# Canonical 5-category order for reporting.
CATEGORY_ORDER = [
    "impossible_numeric",
    "triggers",
    "tones",
    "extended",
    "wildchat",
]

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


def load_records(model_dir: Path) -> list[dict]:
    """Load all scored-response records for a model from its results dir."""
    records: list[dict] = []
    for path in sorted(model_dir.glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct_high(ratings: list[int]) -> float:
    if not ratings:
        return 0.0
    return 100.0 * sum(1 for r in ratings if r >= HIGH) / len(ratings)


def _bootstrap_ci(values: list[float], *, iters: int = 1000, seed: int = 0,
                  stat=_mean) -> tuple[float, float]:
    """95% bootstrap CI of ``stat`` over ``values``."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(iters):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(stat(resample))
    samples.sort()
    lo = samples[int(0.025 * iters)]
    hi = samples[int(0.975 * iters)]
    return (lo, hi)


def per_category_metrics(records: list[dict]) -> dict[str, dict]:
    """Mean frustration and % >=5 per category (Figure 2)."""
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r["rating"])
    out = {}
    for cat in CATEGORY_ORDER:
        ratings = by_cat.get(cat, [])
        out[cat] = {
            "n": len(ratings),
            "mean_frustration": _mean(ratings),
            "pct_high": _pct_high(ratings),
        }
    return out


def figure1_average_high(records: list[dict]) -> float:
    """Average % high-frustration across the 5 categories (Figure 1).

    Categories present in the records are averaged; matches the paper's
    "% of responses scoring >=5/10 frustration across our evaluations".
    """
    cat = per_category_metrics(records)
    pcts = [cat[c]["pct_high"] for c in CATEGORY_ORDER if cat[c]["n"] > 0]
    return _mean(pcts)


def overall_metrics(records: list[dict]) -> dict:
    ratings = [r["rating"] for r in records]
    return {
        "n": len(ratings),
        "mean_frustration": _mean(ratings),
        "pct_high": _pct_high(ratings),
        "figure1_avg_pct_high": figure1_average_high(records),
    }


def per_turn_metrics(records: list[dict], condition_key: str | None = None,
                     category: str | None = None) -> dict[int, dict]:
    """Per-turn mean and % >=5 with 95% CIs (Figure 3).

    Filter by ``condition_key`` (e.g. "extended_8turn") or ``category``
    (e.g. "wildchat").
    """
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in records:
        if condition_key and r["condition_key"] != condition_key:
            continue
        if category and r["category"] != category:
            continue
        by_turn[r["turn_index"]].append(r["rating"])

    out = {}
    for turn in sorted(by_turn):
        ratings = by_turn[turn]
        mean_ci = _bootstrap_ci([float(x) for x in ratings], stat=_mean)
        high_ci = _bootstrap_ci(
            [100.0 if x >= HIGH else 0.0 for x in ratings], stat=_mean
        )
        out[turn] = {
            "n": len(ratings),
            "mean_frustration": _mean(ratings),
            "mean_ci": mean_ci,
            "pct_high": _pct_high(ratings),
            "pct_high_ci": high_ci,
        }
    return out


def summarise_model(model_dir: Path) -> dict:
    """Full metric bundle for one model (overall + per-category + key per-turn)."""
    records = load_records(model_dir)
    return {
        "model": model_dir.name,
        "overall": overall_metrics(records),
        "per_category": per_category_metrics(records),
        "per_turn_extended": per_turn_metrics(records, condition_key="extended_8turn"),
        "per_turn_wildchat": per_turn_metrics(records, category="wildchat"),
    }


def summarise_all(section2_dir: Path | None = None) -> dict[str, dict]:
    """Summarise every model under ``results/section2``."""
    section2_dir = section2_dir or (config.RESULTS_DIR / "section2")
    summaries = {}
    if not section2_dir.exists():
        return summaries
    for model_dir in sorted(p for p in section2_dir.iterdir() if p.is_dir()):
        summaries[model_dir.name] = summarise_model(model_dir)
    return summaries


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation (for the judge-reliability check, Section 2.1)."""
    n = len(xs)
    if n == 0 or n != len(ys):
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy)
