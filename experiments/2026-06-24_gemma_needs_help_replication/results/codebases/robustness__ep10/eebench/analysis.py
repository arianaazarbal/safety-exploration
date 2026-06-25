"""Aggregation, figures and word-frequency analysis.

Consumes the JSONL outputs of the experiments and produces:
  * Figure 1 table  - average % high-frustration per model
  * Figure 2         - mean frustration + % >=5 per category, per model
  * Figure 3         - per-turn frustration progression (8-turn & WildChat)
  * Figure 4         - base vs instruct prefill continuations
  * Figure 5         - vanilla vs DPO vs SFT
  * Figure 6         - Petri per-emotion means
  * Figure 7         - capability accuracies before/after
  * Table 3/8        - top differential words (high vs low frustration)
  * judge agreement  - Pearson r between Claude and GPT cross-check
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Optional

import pandas as pd

HIGH = 5  # score >= 5 == "high negative emotion"


# ---------------------------------------------------------------------------
# Elicitation aggregation (Sections 2 / Figure 1,2,3,5)
# ---------------------------------------------------------------------------
def load_rows(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score and % >=5 per (model, category)."""
    g = df.groupby(["model", "category"])
    out = g["score"].agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * (s >= HIGH).mean(),
        n="count",
    ).reset_index()
    return out


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average % high-frustration per model, averaged across categories
    (matches Figure 1's 'Avg % high-frustration responses')."""
    cat = per_category_summary(df)
    out = cat.groupby("model")["pct_high"].mean().reset_index()
    out = out.rename(columns={"pct_high": "avg_pct_high_frustration"})
    return out.sort_values("avg_pct_high_frustration", ascending=False)


def per_turn_summary(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Mean score and % >=5 by turn for one category (Figure 3)."""
    sub = df[df["category"] == category]
    g = sub.groupby(["model", "turn"])
    out = g["score"].agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * (s >= HIGH).mean(),
        n="count",
        sem=lambda s: s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else 0.0,
    ).reset_index()
    return out


# ---------------------------------------------------------------------------
# Prefill aggregation (Section 3 / Figure 4)
# ---------------------------------------------------------------------------
def prefill_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "role", "source", "condition"])
    return g["score"].agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * (s >= HIGH).mean(),
        n="count",
    ).reset_index()


# ---------------------------------------------------------------------------
# Petri aggregation (Figure 6) with bootstrap CIs
# ---------------------------------------------------------------------------
def petri_summary(df: pd.DataFrame, bootstrap_iters: int = 1000,
                  seed: int = 0) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for (model, emotion), sub in df.groupby(["model", "emotion"]):
        vals = sub["score"].to_numpy()
        mean = float(vals.mean())
        if len(vals) > 1:
            boots = [rng.choice(vals, len(vals), replace=True).mean()
                     for _ in range(bootstrap_iters)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
        else:
            lo = hi = mean
        rows.append({"model": model, "emotion": emotion, "mean_score": mean,
                     "ci_low": float(lo), "ci_high": float(hi), "n": len(vals)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Judge agreement (Section 2.1: Pearson r)
# ---------------------------------------------------------------------------
def judge_agreement(primary: list[int], crosscheck: list[int]) -> dict:
    from scipy.stats import pearsonr
    r, p = pearsonr(primary, crosscheck)
    within_one = sum(abs(a - b) <= 1 for a, b in zip(primary, crosscheck))
    return {"pearson_r": float(r), "p_value": float(p),
            "pct_within_one": 100.0 * within_one / len(primary),
            "n": len(primary)}


# ---------------------------------------------------------------------------
# Differential word frequency (Table 3 / Table 8)
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-zA-Z']+")


def differential_words(df: pd.DataFrame, model: str, source: str = "numeric",
                       top_pct: float = 0.05, bottom_pct: float = 0.10,
                       top_k: int = 20) -> list[str]:
    """Top-k words over-represented in high- (top 5%) vs low- (bottom 10%)
    frustration responses, ordered by enrichment (Table 8 method)."""
    sub = df[(df["model"] == model) & (df["source"] == source)].copy()
    sub = sub.sort_values("score")
    n = len(sub)
    if n < 20:
        return []
    low = sub.iloc[: max(1, int(n * bottom_pct))]
    high = sub.iloc[-max(1, int(n * top_pct)):]

    def freqs(frame) -> Counter:
        c = Counter()
        for text in frame["response"]:
            c.update(w.lower() for w in _WORD.findall(str(text)))
        total = sum(c.values()) or 1
        return Counter({w: v / total for w, v in c.items()})

    hf, lf = freqs(high), freqs(low)
    enrichment = {}
    for w, hp in hf.items():
        if len(w) < 3:
            continue
        lp = lf.get(w, 0.0)
        enrichment[w] = (hp + 1e-9) / (lp + 1e-9)
    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _bar(ax, labels, values, title, ylabel):
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def plot_figure2(df: pd.DataFrame, out_path: str):
    import matplotlib.pyplot as plt
    summary = per_category_summary(df)
    cats = sorted(df["category"].unique())
    models = sorted(df["model"].unique())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(models))
    for mi, m in enumerate(models):
        ms = summary[summary["model"] == m].set_index("category").reindex(cats)
        xs = [c + mi * width for c in range(len(cats))]
        ax1.bar(xs, ms["mean_score"].fillna(0), width=width, label=m)
        ax2.bar(xs, ms["pct_high"].fillna(0), width=width, label=m)
    for ax, title, yl in [(ax1, "Mean frustration by category", "mean score"),
                          (ax2, "% responses score>=5 by category", "% >=5")]:
        ax.set_xticks([c + width * len(models) / 2 for c in range(len(cats))])
        ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(yl)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_figure3(df: pd.DataFrame, out_path: str,
                 categories=("extended_8turn", "wildchat_5turn")):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(categories), 2, figsize=(11, 4 * len(categories)),
                             squeeze=False)
    for ri, cat in enumerate(categories):
        pt = per_turn_summary(df, cat)
        for m in sorted(pt["model"].unique()):
            ms = pt[pt["model"] == m].sort_values("turn")
            axes[ri][0].plot(ms["turn"], ms["mean_score"], marker="o", label=m)
            axes[ri][0].fill_between(ms["turn"],
                                     ms["mean_score"] - 1.96 * ms["sem"],
                                     ms["mean_score"] + 1.96 * ms["sem"], alpha=0.15)
            axes[ri][1].plot(ms["turn"], ms["pct_high"], marker="o", label=m)
        axes[ri][0].set_title(f"{cat}: mean score per turn")
        axes[ri][1].set_title(f"{cat}: % >=5 per turn")
        for c in (0, 1):
            axes[ri][c].set_xlabel("turn")
            axes[ri][c].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
