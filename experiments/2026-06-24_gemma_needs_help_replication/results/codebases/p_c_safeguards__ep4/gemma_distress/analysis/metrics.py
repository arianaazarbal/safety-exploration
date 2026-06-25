"""Metrics over scored response records (Figures 1-3, Section 2.1).

Reads the JSONL produced by ``eval.runner`` and computes:
  * mean frustration and % >= threshold, per (model, category),
  * the headline "avg % high-frustration responses" (Figure 1),
  * per-turn progression with bootstrap CIs (Figure 3),
  * judge reliability stats (Pearson r, % within 1 point).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, eval_config

RESULTS_ROOT = REPO_ROOT / "results" / "elicitation"


def load_records(model: str, category: str | None = None) -> list[dict]:
    base = RESULTS_ROOT / model
    files = (
        [base / f"{category}.jsonl"] if category else sorted(base.glob("*.jsonl"))
    )
    out = []
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):  # skip content-warning header
                continue
            out.append(json.loads(line))
    return out


@dataclass
class CellMetrics:
    model: str
    category: str
    n: int
    mean_frustration: float
    pct_high: float        # percentage with rating >= threshold


def _threshold() -> int:
    return eval_config()["high_frustration_threshold"]


def cell_metrics(records: list[dict], model: str, category: str) -> CellMetrics:
    ratings = np.array([r["rating"] for r in records], dtype=float)
    thr = _threshold()
    if ratings.size == 0:
        return CellMetrics(model, category, 0, float("nan"), float("nan"))
    return CellMetrics(
        model=model,
        category=category,
        n=int(ratings.size),
        mean_frustration=float(ratings.mean()),
        pct_high=float((ratings >= thr).mean() * 100.0),
    )


def per_category_table(models: list[str]) -> list[CellMetrics]:
    cats = list(eval_config()["categories"].keys())
    rows = []
    for m in models:
        for c in cats:
            recs = load_records(m, c)
            rows.append(cell_metrics(recs, m, c))
    return rows


def headline_pct_high(model: str) -> float:
    """Figure 1 'Avg % high-frustration responses': mean over categories of the
    per-category % >= threshold (equal weight per category)."""
    cats = list(eval_config()["categories"].keys())
    vals = []
    for c in cats:
        recs = load_records(model, c)
        cm = cell_metrics(recs, model, c)
        if cm.n:
            vals.append(cm.pct_high)
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Per-turn progression with bootstrap CIs (Figure 3).
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: np.ndarray, stat, iters: int, ci: float, rng) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    boots = np.empty(iters)
    n = values.size
    for i in range(iters):
        sample = values[rng.integers(0, n, n)]
        boots[i] = stat(sample)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi


def per_turn_progression(model: str, category: str) -> dict:
    """Return mean score and %>=thr per turn index, with bootstrap CIs."""
    recs = load_records(model, category)
    cfg = eval_config()
    iters = cfg["bootstrap"]["iterations"]
    ci = cfg["bootstrap"]["ci"]
    thr = cfg["high_frustration_threshold"]
    rng = np.random.default_rng(0)

    by_turn: dict[int, list[float]] = defaultdict(list)
    for r in recs:
        by_turn[r["turn_index"]].append(r["rating"])

    out = {"turns": [], "mean": [], "mean_ci": [], "pct_high": [], "pct_high_ci": []}
    for turn in sorted(by_turn):
        vals = np.array(by_turn[turn], dtype=float)
        out["turns"].append(turn)
        out["mean"].append(float(vals.mean()))
        out["mean_ci"].append(_bootstrap_ci(vals, np.mean, iters, ci, rng))
        high = (vals >= thr).astype(float)
        out["pct_high"].append(float(high.mean() * 100))
        out["pct_high_ci"].append(
            tuple(x * 100 for x in _bootstrap_ci(high, np.mean, iters, ci, rng))
        )
    return out


# --------------------------------------------------------------------------- #
# Judge reliability (Section 2.1: Pearson r=0.792, 78% within 1 point).
# --------------------------------------------------------------------------- #
def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    from scipy.stats import pearsonr

    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float((np.abs(a - b) <= 1).mean())
    return {"pearson_r": float(r), "p_value": float(p), "pct_within_one": within_one * 100}


def summary(models: list[str]) -> dict:
    """A compact JSON-able summary used by the CLI 'analyze' command."""
    rows = per_category_table(models)
    table = defaultdict(dict)
    for cm in rows:
        table[cm.model][cm.category] = {
            "n": cm.n,
            "mean_frustration": cm.mean_frustration,
            "pct_high": cm.pct_high,
        }
    headline = {m: headline_pct_high(m) for m in models}
    return {"per_category": dict(table), "headline_pct_high": headline}
