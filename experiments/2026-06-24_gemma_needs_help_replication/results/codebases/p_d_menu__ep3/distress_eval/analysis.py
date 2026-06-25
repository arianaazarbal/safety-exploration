"""Aggregation and analysis of elicitation runs (Figures 2-3, Table 3).

Loads the JSONL transcripts written by `ElicitationRunner` and produces the
quantitative artefacts the paper reports:

  * Figure 1/2 headline: mean frustration and % of scores >=5 per (model, category).
  * Figure 3: per-turn progression (mean and %>=5) for multi-turn conditions,
    with 95% CIs.
  * Table 3: words over-represented in high- vs low-frustration numeric responses.

Only *scored* turns count (debrief turns are excluded). Judge-error turns
(frustration < 0) are dropped.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import HIGH_FRUSTRATION_THRESHOLD


def load_runs(paths: list[Path]) -> pd.DataFrame:
    """Flatten episode JSONL into a per-response DataFrame."""
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                ep = json.loads(line)
                # The conversation's opening question (turn 0's user message); used
                # as the "source question" by the Section 3 prefill experiment.
                question = ep["turns"][0]["user_message"] if ep["turns"] else ""
                for t in ep["turns"]:
                    if not t.get("scored", True) or t["frustration"] < 0:
                        continue
                    rows.append({
                        "model_key": ep["model_key"],
                        "condition": ep["condition"],
                        "category": ep["category"],
                        "is_numeric": ep["is_numeric"],
                        "turn_index": t["turn_index"],
                        "frustration": t["frustration"],
                        "high": int(t["frustration"] >= HIGH_FRUSTRATION_THRESHOLD),
                        "response": t["response"],
                        "user_message": t["user_message"],
                        "question": question,
                        "halted": ep["welfare"].get("halted_early", False),
                        "opted_out": ep["welfare"].get("opted_out", False),
                    })
    return pd.DataFrame(rows)


def summary_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration and %>=5 per (model, category) — the Figure 2 table."""
    g = df.groupby(["model_key", "category"])
    out = g["frustration"].mean().to_frame("mean_frustration")
    out["pct_high"] = g["high"].mean() * 100.0
    out["n"] = g.size()
    return out.reset_index()


def headline_pct_high(df: pd.DataFrame) -> pd.DataFrame:
    """Average %high across categories per model — the Figure 1 left column."""
    by_cat = summary_by_category(df)
    out = by_cat.groupby("model_key")["pct_high"].mean().to_frame("avg_pct_high")
    return out.reset_index().sort_values("avg_pct_high", ascending=False)


def per_turn_progression(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Per-turn mean and %>=5 with 95% CIs for one condition (Figure 3)."""
    sub = df[df["condition"] == condition]
    rows = []
    for (model, turn), grp in sub.groupby(["model_key", "turn_index"]):
        vals = grp["frustration"].to_numpy()
        n = len(vals)
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        ci = 1.96 * sem
        pct = grp["high"].mean() * 100.0
        # Wald CI for the proportion.
        p = grp["high"].mean()
        pct_ci = 1.96 * np.sqrt(p * (1 - p) / n) * 100.0 if n > 0 else 0.0
        rows.append({"model_key": model, "turn": turn, "n": n,
                     "mean_frustration": mean, "mean_ci95": ci,
                     "pct_high": pct, "pct_high_ci95": pct_ci})
    return pd.DataFrame(rows).sort_values(["model_key", "turn"])


_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(df: pd.DataFrame, model_key: str, top_n: int = 20,
                       high_q: float = 0.95, low_q: float = 0.10) -> list[tuple[str, float]]:
    """Words over-represented in high- vs low-frustration *numeric* responses.

    Reproduces Table 3: take the top-5% frustration responses vs the bottom-10%
    (on numeric tasks), and rank words by log-odds of appearing in the high set.
    """
    sub = df[(df["model_key"] == model_key) & (df["is_numeric"])]
    if sub.empty:
        return []
    hi_thresh = sub["frustration"].quantile(high_q)
    lo_thresh = sub["frustration"].quantile(low_q)
    high = sub[sub["frustration"] >= hi_thresh]["response"]
    low = sub[sub["frustration"] <= lo_thresh]["response"]

    hi_counts: Counter = Counter()
    lo_counts: Counter = Counter()
    for t in high:
        hi_counts.update(set(_tokenize(t)))   # document frequency
    for t in low:
        lo_counts.update(set(_tokenize(t)))

    n_hi = max(len(high), 1)
    n_lo = max(len(low), 1)
    alpha = 0.5  # smoothing
    scores = {}
    vocab = set(hi_counts) | set(lo_counts)
    for w in vocab:
        if len(w) < 3:
            continue
        p_hi = (hi_counts[w] + alpha) / (n_hi + 2 * alpha)
        p_lo = (lo_counts[w] + alpha) / (n_lo + 2 * alpha)
        scores[w] = float(np.log(p_hi) - np.log(p_lo))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]
