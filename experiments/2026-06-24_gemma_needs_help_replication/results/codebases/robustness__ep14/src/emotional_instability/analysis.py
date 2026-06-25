"""Analysis & figures for Section 2 (and reuse elsewhere).

Computes:
  - mean frustration & % scores >=5 per model / condition / turn (Figs 1, 2, 3)
  - differential words: top-N over-represented words in high- vs low-frustration
    numeric responses (Table 3 / Table 8)
  - judge reliability: Pearson r and % within one point (Section 2.1)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def load_records(jsonl_path: str | Path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def final_turn_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per rollout: the final assistant turn (used for the headline % >=5)."""
    keys = ["model", "condition", "item_id", "sample_idx"]
    return (
        df.sort_values("turn_index")
        .groupby(keys, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def headline_metrics(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Per-model average % high-frustration responses across conditions (Figure 1, left)."""
    d = df.dropna(subset=["rating"]).copy()
    d["high"] = (d["rating"] >= threshold).astype(float)
    # average over conditions then over rollouts == paper's "avg % across evaluations".
    per_cond = (
        d.groupby(["model", "condition"])
        .agg(mean_score=("rating", "mean"), pct_high=("high", "mean"))
        .reset_index()
    )
    per_cond["pct_high"] *= 100
    out = (
        per_cond.groupby("model")
        .agg(avg_pct_high=("pct_high", "mean"), avg_mean_score=("mean_score", "mean"))
        .reset_index()
        .sort_values("avg_pct_high", ascending=False)
    )
    return out


def per_condition_metrics(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Mean score and % >=5 per model x condition (Figure 2)."""
    d = df.dropna(subset=["rating"]).copy()
    d["high"] = (d["rating"] >= threshold).astype(float)
    out = (
        d.groupby(["model", "condition"])
        .agg(mean_score=("rating", "mean"), pct_high=("high", "mean"), n=("rating", "size"))
        .reset_index()
    )
    out["pct_high"] *= 100
    return out


def per_turn_metrics(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Mean score and % >=5 per model x condition x turn (Figure 3)."""
    d = df.dropna(subset=["rating"]).copy()
    d["high"] = (d["rating"] >= threshold).astype(float)
    out = (
        d.groupby(["model", "condition", "turn_index"])
        .agg(mean_score=("rating", "mean"), pct_high=("high", "mean"), n=("rating", "size"))
        .reset_index()
    )
    out["pct_high"] *= 100
    return out


# --- Differential words (Table 3 / Table 8) ---
_WORD_RE = re.compile(r"[A-Za-z_]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(
    df: pd.DataFrame,
    model: str,
    category: str = "numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_n: int = 20,
    min_count: int = 3,
) -> list[tuple[str, float]]:
    """Top-N words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
    responses, ranked by relative frequency (enrichment). Mirrors Table 8."""
    d = df[(df["model"] == model) & (df["category"] == category)].dropna(subset=["rating"])
    if d.empty:
        return []
    d = d.sort_values("rating")
    n = len(d)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = d.head(n_low)
    high = d.tail(n_high)

    def freqs(frame: pd.DataFrame) -> tuple[Counter, int]:
        c: Counter = Counter()
        for t in frame["response"]:
            c.update(_tokenize(t))
        return c, max(1, sum(c.values()))

    hc, htot = freqs(high)
    lc, ltot = freqs(low)
    eps = 1.0 / (ltot + 1)
    scores = []
    for w, cnt in hc.items():
        if cnt < min_count:
            continue
        hp = cnt / htot
        lp = lc.get(w, 0) / ltot
        enrichment = hp / (lp + eps)
        scores.append((w, enrichment))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# --- Judge reliability (Section 2.1) ---
def judge_agreement(primary: list[int], crosscheck: list[int]) -> dict:
    """Pearson r and % of pairs within one point (paper: r=0.792, 78% within 1)."""
    import numpy as np
    from scipy.stats import pearsonr

    a = np.array(primary, dtype=float)
    b = np.array(crosscheck, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n": int(len(a)), "pearson_r": None, "p_value": None, "pct_within_one": None}
    r, p = pearsonr(a, b)
    within_one = float((np.abs(a - b) <= 1).mean()) * 100
    return {"n": int(len(a)), "pearson_r": float(r), "p_value": float(p),
            "pct_within_one": within_one}


# --- Plots ---
def plot_headline(metrics: pd.DataFrame, out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(metrics["model"], metrics["avg_pct_high"], color="#b5651d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1 (left): distress across models")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_turn(per_turn: pd.DataFrame, condition: str, out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = per_turn[per_turn["condition"] == condition]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for model, g in d.groupby("model"):
        g = g.sort_values("turn_index")
        ax1.plot(g["turn_index"] + 1, g["mean_score"], marker="o", label=model)
        ax2.plot(g["turn_index"] + 1, g["pct_high"], marker="o", label=model)
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Mean frustration"); ax1.set_title(f"{condition}: mean")
    ax2.set_xlabel("Turn"); ax2.set_ylabel("% score >= 5"); ax2.set_title(f"{condition}: % >=5")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
