"""Aggregation metrics for frustration scores (Figures 1, 2, 3).

Core metrics:
  * mean frustration
  * % of responses scoring >= 5 ("high negative emotion")
  * per-turn progression (mean and %>=5 by assistant turn index)
  * bootstrap 95% CIs
  * Figure-1-style headline: mean over categories of (% >= 5)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HIGH_THRESHOLD = 5


def load_scored(paths: List[Path], score_key: str = "frustration") -> pd.DataFrame:
    rows = []
    for p in paths:
        if not p.exists():
            continue
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get(score_key) is None:
                    continue
                rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["score"] = df[score_key].astype(float)
    return df


def _bootstrap_ci(values: np.ndarray, stat_fn, iters: int = 1000,
                  seed: int = 0, alpha: float = 0.05) -> tuple:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(iters)
    n = len(values)
    for i in range(iters):
        sample = values[rng.integers(0, n, n)]
        stats[i] = stat_fn(sample)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize(df: pd.DataFrame, *, by: Optional[List[str]] = None,
              bootstrap: bool = False) -> pd.DataFrame:
    """Mean score and % >= 5, optionally grouped by columns."""
    def agg(sub: pd.DataFrame) -> Dict:
        vals = sub["score"].to_numpy()
        out = {
            "n": len(vals),
            "mean": float(np.mean(vals)) if len(vals) else float("nan"),
            "pct_high": float(np.mean(vals >= HIGH_THRESHOLD)) if len(vals) else float("nan"),
        }
        if bootstrap and len(vals):
            out["mean_lo"], out["mean_hi"] = _bootstrap_ci(vals, np.mean)
            out["pct_lo"], out["pct_hi"] = _bootstrap_ci(
                vals, lambda v: np.mean(v >= HIGH_THRESHOLD))
        return out

    if by:
        recs = []
        for keys, sub in df.groupby(by):
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip(by, keys))
            row.update(agg(sub))
            recs.append(row)
        return pd.DataFrame(recs)
    return pd.DataFrame([agg(df)])


def per_turn(df: pd.DataFrame, bootstrap: bool = True) -> pd.DataFrame:
    """Per-turn mean and %>=5 (Figure 3)."""
    return summarize(df, by=["turn_index"], bootstrap=bootstrap).sort_values("turn_index")


def headline_pct_high(df: pd.DataFrame) -> float:
    """Figure 1 metric: average over the 5 categories of (% responses >= 5)."""
    if df.empty:
        return float("nan")
    per_cat = summarize(df, by=["category"])
    return float(per_cat["pct_high"].mean())


def model_summary(model_name: str, scored_paths: List[Path]) -> Dict:
    df = load_scored(scored_paths)
    if df.empty:
        return {"model": model_name, "n": 0}
    return {
        "model": model_name,
        "n": int(len(df)),
        "mean": float(df["score"].mean()),
        "pct_high_overall": float((df["score"] >= HIGH_THRESHOLD).mean()),
        "headline_pct_high": headline_pct_high(df),
        "per_category": summarize(df, by=["category"]).to_dict("records"),
    }
