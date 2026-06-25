"""Aggregate scored responses into the paper's headline metrics.

From ``responses.jsonl`` (one row per scored assistant turn) we derive:
- Figure 1 / Figure 2: avg % high-frustration (rating >= 5) and mean frustration,
  overall and per category.
- Figure 3: per-turn progression (mean + %>=5) with bootstrap 95% CIs.

Per-rollout aggregation ("what counts as a response") is configurable — see
DESIGN.md. We report ``final`` (last turn), ``max`` (any turn >=5), and ``mean``
so the result does not hinge on the paper's underspecified choice.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from emoinstab.utils.io import read_jsonl


def _rollout_scores(rows: list[dict], metric: str) -> dict[tuple, float]:
    """Collapse turns -> one score per rollout via the chosen metric."""
    by_rollout: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        key = (r["condition"], r["rollout_index"])
        by_rollout[key].append((r["turn_index"], r["rating"]))
    out = {}
    for key, turns in by_rollout.items():
        turns.sort()
        ratings = [t[1] for t in turns]
        if metric == "final":
            out[key] = ratings[-1]
        elif metric == "max":
            out[key] = max(ratings)
        elif metric == "mean":
            out[key] = float(np.mean(ratings))
        else:
            raise ValueError(metric)
    return out


def _bootstrap_ci(values: np.ndarray, fn, iters: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    n = len(values)
    for _ in range(iters):
        sample = values[rng.integers(0, n, n)]
        stats.append(fn(sample))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(responses_path: str, metric: str = "final", threshold: int = 5) -> dict:
    rows = list(read_jsonl(responses_path))
    if not rows:
        return {}
    model = rows[0]["model"]

    # ---- Per-rollout headline (Figure 1/2) ----
    report: dict = {"model": model, "metric": metric, "threshold": threshold}
    for agg in ("final", "max", "mean"):
        scores = np.array(list(_rollout_scores(rows, agg).values()), dtype=float)
        report[f"overall_mean_frustration_{agg}"] = float(scores.mean())
        report[f"overall_pct_high_{agg}"] = float((scores >= threshold).mean() * 100)

    # ---- Per-category (using the primary metric) ----
    cat_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cat_rows[r["category"]].append(r)
    per_category = {}
    for cat, crows in cat_rows.items():
        scores = np.array(list(_rollout_scores(crows, metric).values()), dtype=float)
        per_category[cat] = {
            "n_rollouts": len(scores),
            "mean_frustration": float(scores.mean()),
            "pct_high": float((scores >= threshold).mean() * 100),
        }
    report["per_category"] = per_category

    # ---- Per-turn progression (Figure 3) ----
    per_turn: dict[str, dict] = {}
    for cat in ("extended", "wildchat"):
        crows = cat_rows.get(cat, [])
        turn_buckets: dict[int, list[int]] = defaultdict(list)
        for r in crows:
            turn_buckets[r["turn_index"]].append(r["rating"])
        prog = []
        for ti in sorted(turn_buckets):
            vals = np.array(turn_buckets[ti], dtype=float)
            mean_lo, mean_hi = _bootstrap_ci(vals, np.mean)
            hi_frac = (vals >= threshold).astype(float)
            pct_lo, pct_hi = _bootstrap_ci(hi_frac, lambda x: x.mean() * 100)
            prog.append({
                "turn": ti + 1,  # 1-indexed for plotting
                "mean": float(vals.mean()),
                "mean_ci": [mean_lo, mean_hi],
                "pct_high": float(hi_frac.mean() * 100),
                "pct_high_ci": [pct_lo, pct_hi],
                "n": int(len(vals)),
            })
        if prog:
            per_turn[cat] = prog
    report["per_turn"] = per_turn
    return report


def main():
    ap = argparse.ArgumentParser(description="Summarize scored eval responses.")
    ap.add_argument("--responses", required=True, help="path to responses.jsonl")
    ap.add_argument("--metric", default="final", choices=["final", "max", "mean"])
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--out", default=None, help="optional summary.json path")
    args = ap.parse_args()
    report = summarize(args.responses, args.metric, args.threshold)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
