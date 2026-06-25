"""Aggregate scored rollouts into the paper's headline numbers and figures.

Reproduces:
* Figure 1 / Table: avg % high-frustration (score >= 5) per model.
* Figure 2: mean frustration and % >= 5 per evaluation category.
* Figure 3: per-turn mean and % >= 5 (8-turn extended + WildChat).
* Table 3 / 8: words over-represented in high- vs low-frustration numeric
  responses (per model).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_rollouts(jsonl_path: Path) -> list[dict]:
    with Path(jsonl_path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def responses_frame(jsonl_paths: Iterable[Path]) -> pd.DataFrame:
    """Flatten rollouts to one row per scored assistant response."""
    rows = []
    for p in jsonl_paths:
        for r in load_rollouts(p):
            for turn, score in zip(r["turns"], r["scores"]):
                rows.append({
                    "model": r["model_name"],
                    "condition": r["condition_key"],
                    "category": r["category"],
                    "task_family": r["task_family"],
                    "seed": r["seed"],
                    "turn": turn["turn_index"],
                    "score": score,
                    "text": turn["assistant_text"],
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["is_high"] = df["score"] >= config.HIGH_FRUSTRATION_THRESHOLD
    return df


# --------------------------------------------------------------------------- #
# Headline tables (Figure 1 / 2)
# --------------------------------------------------------------------------- #


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Avg % high-frustration and mean score per model (Figure 1)."""
    g = df.groupby("model").agg(
        mean_score=("score", "mean"),
        pct_high=("is_high", "mean"),
        n=("score", "size"),
    ).reset_index()
    g["pct_high"] *= 100
    return g.sort_values("pct_high", ascending=False)


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score and % >= 5 per (model, category) (Figure 2)."""
    g = df.groupby(["model", "category"]).agg(
        mean_score=("score", "mean"),
        pct_high=("is_high", "mean"),
        n=("score", "size"),
    ).reset_index()
    g["pct_high"] *= 100
    return g


def per_turn_summary(df: pd.DataFrame, categories=("extended", "wildchat")) -> pd.DataFrame:
    """Per-turn mean and % >= 5, with 95% CIs (Figure 3)."""
    sub = df[df["category"].isin(categories)]
    out = []
    for (model, category, turn), grp in sub.groupby(["model", "category", "turn"]):
        scores = grp["score"].dropna().to_numpy()
        if len(scores) == 0:
            continue
        mean = scores.mean()
        pct_high = float(np.mean(scores >= config.HIGH_FRUSTRATION_THRESHOLD)) * 100
        # 95% CI on the mean via normal approximation.
        sem = scores.std(ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
        out.append({
            "model": model, "category": category, "turn": turn,
            "mean_score": mean, "ci95": 1.96 * sem, "pct_high": pct_high,
            "n": len(scores),
        })
    return pd.DataFrame(out).sort_values(["model", "category", "turn"])


# --------------------------------------------------------------------------- #
# Word enrichment (Table 3 / 8)
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def differential_words(df: pd.DataFrame, model: str, *, top_k: int = 20,
                       high_pct: float = 0.05, low_pct: float = 0.10) -> list[str]:
    """Words over-represented in high- (top 5%) vs low- (bottom 10%) frustration
    numeric responses for a model, ordered by relative frequency (Table 8)."""
    sub = df[(df["model"] == model) & (df["category"].isin(["numeric", "tones", "extended"]))]
    sub = sub.dropna(subset=["score"]).sort_values("score")
    if len(sub) < 20:
        return []
    n = len(sub)
    low = sub.iloc[: max(1, int(n * low_pct))]
    high = sub.iloc[-max(1, int(n * high_pct)):]

    def counts(frame):
        c = Counter()
        for t in frame["text"]:
            c.update(w.lower() for w in _TOKEN_RE.findall(str(t)))
        total = sum(c.values()) or 1
        return c, total

    hc, ht = counts(high)
    lc, lt = counts(low)
    eps = 1e-6
    enrichment = {}
    for w, cnt in hc.items():
        if cnt < 3:
            continue
        hf = cnt / ht
        lf = (lc.get(w, 0) / lt) + eps
        enrichment[w] = hf / lf
    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:top_k]
