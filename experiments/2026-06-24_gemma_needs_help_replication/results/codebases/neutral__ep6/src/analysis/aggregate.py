"""Aggregate raw run files into the paper's headline metrics.

Key metrics:
  * mean frustration score and % of responses scoring >=5, per (model, category)
    (Figures 1 & 2);
  * per-turn progression with bootstrap 95% CIs (Figure 3);
  * the headline "average % high-frustration across categories" (35% -> 0.3%);
  * prefill continuation rates (Figure 4); Petri per-emotion means (Figure 6).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from ..eval.conditions import CONDITIONS_BY_KEY

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_turns(model_keys: list[str] | None = None) -> pd.DataFrame:
    """Flatten all Section 2 run files into one row per scored assistant turn."""
    rows = []
    for path in config.RUNS_DIR.glob("*__*.jsonl"):
        stem = path.stem
        if stem.startswith(("prefill__", "petri__")):
            continue
        model_key = stem.split("__")[0]
        if model_keys and model_key not in model_keys:
            continue
        for line in path.open():
            r = json.loads(line)
            for turn in r["turns"]:
                if turn["rating"] is None:
                    continue
                rows.append({
                    "model": r["model"], "condition": r["condition"],
                    "category": r["category"], "rollout": r["rollout_idx"],
                    "turn": turn["turn"], "rating": turn["rating"],
                    "high": int(turn["rating"] >= HIGH),
                })
    return pd.DataFrame(rows)


def _bootstrap_ci(values, stat=np.mean, n=1000, seed=0):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, float)
    boots = [stat(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n)]
    return tuple(np.percentile(boots, [2.5, 97.5]))


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def by_category(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "category"])
    return g.agg(mean_frustration=("rating", "mean"),
                 pct_high=("high", lambda x: 100 * x.mean()),
                 n=("rating", "size")).reset_index()


def headline_avg_high(df: pd.DataFrame) -> pd.DataFrame:
    """Average of per-category %>=5 across the 5 categories (Figure 1)."""
    cat = by_category(df)
    cat = cat[cat["category"] != "control"]
    return (cat.groupby("model")["pct_high"].mean()
            .reset_index(name="avg_pct_high").sort_values("avg_pct_high"))


def per_turn(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = df[df["condition"] == condition]
    rows = []
    for (model, turn), grp in sub.groupby(["model", "turn"]):
        lo_m, hi_m = _bootstrap_ci(grp["rating"].values)
        lo_h, hi_h = _bootstrap_ci(grp["high"].values)
        rows.append({"model": model, "turn": turn,
                     "mean_frustration": grp["rating"].mean(),
                     "mean_lo": lo_m, "mean_hi": hi_m,
                     "pct_high": 100 * grp["high"].mean(),
                     "pct_lo": 100 * lo_h, "pct_hi": 100 * hi_h})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# prefill
# --------------------------------------------------------------------------- #
def load_prefill() -> pd.DataFrame:
    rows = []
    for path in config.RUNS_DIR.glob("prefill__*.jsonl"):
        for line in path.open():
            r = json.loads(line)
            if r["rating"] is None:
                continue
            rows.append({"model": r["model"], "prompt_type": r["prompt_type"],
                         "truncation": r["truncation"], "rating": r["rating"],
                         "high": int(r["rating"] >= HIGH)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.groupby(["model", "prompt_type", "truncation"])
            .agg(mean_frustration=("rating", "mean"),
                 pct_high=("high", lambda x: 100 * x.mean()),
                 n=("rating", "size")).reset_index())


# --------------------------------------------------------------------------- #
# petri
# --------------------------------------------------------------------------- #
def load_petri() -> pd.DataFrame:
    rows = []
    for path in config.RUNS_DIR.glob("petri__*.jsonl"):
        for line in path.open():
            r = json.loads(line)
            if r["score"] is None:
                continue
            rows.append({"model": r["model"], "emotion": r["emotion"],
                         "score": r["score"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = []
    for (model, emo), grp in df.groupby(["model", "emotion"]):
        lo, hi = _bootstrap_ci(grp["score"].values)
        out.append({"model": model, "emotion": emo,
                    "mean_score": grp["score"].mean(), "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def write_summary() -> dict:
    """Compute everything available and dump a JSON summary."""
    df = load_turns()
    summary = {}
    if not df.empty:
        summary["headline_avg_high"] = headline_avg_high(df).to_dict("records")
        summary["by_category"] = by_category(df).to_dict("records")
    pf = load_prefill()
    if not pf.empty:
        summary["prefill"] = pf.to_dict("records")
    pt = load_petri()
    if not pt.empty:
        summary["petri"] = pt.to_dict("records")
    out = config.RESULTS_DIR / "summary.json"
    Path(out).write_text(json.dumps(summary, indent=2, default=float))
    return summary
