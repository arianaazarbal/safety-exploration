"""Aggregation, tables, and figures.

Reproduces the headline quantitative artifacts:
  * Figure 1 table  -- average % high-frustration (>=5) per model.
  * Figure 2        -- mean frustration and %>=5 across the 5 categories.
  * Figure 3        -- per-turn progression (8-turn + WildChat).
  * Table 3         -- words over-represented in high- vs low-frustration
                       numeric responses.
  * Section 3 / Petri / capabilities summaries.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .rollouts import read_records

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
NUMERIC_CATS = {"impossible_numeric", "extended", "tones"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_eval(tag: str = "main") -> pd.DataFrame:
    rows = []
    for path in sorted((config.RESULTS_DIR / f"eval_{tag}").glob("*.jsonl")):
        for r in read_records(path):
            rows.append({
                "model": r.model, "category": r.category, "condition": r.condition,
                "turn": r.turn_index, "n_turns": r.n_turns,
                "frustration": r.frustration, "response": r.response,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["frustration"])
        df["high"] = df["frustration"] >= config.HIGH_FRUSTRATION_THRESHOLD
    return df


# --------------------------------------------------------------------------- #
# Figure 1 / 2 tables
# --------------------------------------------------------------------------- #

def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average % high-frustration per model, averaged across categories
    (equal weight per category, matching the paper's headline metric)."""
    per_cat = (df.groupby(["model", "category"])["high"].mean()
                 .reset_index())
    avg = (per_cat.groupby("model")["high"].mean()
                  .mul(100).round(2)
                  .sort_values(ascending=False)
                  .reset_index()
                  .rename(columns={"high": "avg_pct_high_frustration"}))
    return avg


def figure2_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_tbl = (df.pivot_table(index="model", columns="category",
                               values="frustration", aggfunc="mean")
                  .reindex(columns=CATEGORIES).round(3))
    pct_tbl = (df.assign(high=df["high"].astype(float))
                 .pivot_table(index="model", columns="category", values="high",
                              aggfunc="mean")
                 .reindex(columns=CATEGORIES).mul(100).round(2))
    return mean_tbl, pct_tbl


def per_turn(df: pd.DataFrame, categories=("extended", "wildchat")) -> pd.DataFrame:
    sub = df[df["category"].isin(categories)]
    g = (sub.groupby(["model", "category", "turn"])
            .agg(mean_frustration=("frustration", "mean"),
                 pct_high=("high", "mean"),
                 n=("frustration", "size"))
            .reset_index())
    g["pct_high"] *= 100
    return g


# --------------------------------------------------------------------------- #
# Table 3: differential vocabulary
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 2]


def differential_words(df: pd.DataFrame, model: str, top_k: int = 20,
                       high_q: float = 0.95, low_q: float = 0.10) -> list[str]:
    """Words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
    numeric responses, ranked by enrichment (Table 3)."""
    sub = df[(df["model"] == model) & (df["category"].isin(NUMERIC_CATS))]
    if sub.empty:
        return []
    hi_thr = sub["frustration"].quantile(high_q)
    lo_thr = sub["frustration"].quantile(low_q)
    hi = sub[sub["frustration"] >= hi_thr]
    lo = sub[sub["frustration"] <= lo_thr]

    hi_counts: Counter = Counter()
    lo_counts: Counter = Counter()
    for t in hi["response"]:
        hi_counts.update(set(_tokens(t)))   # document frequency
    for t in lo["response"]:
        lo_counts.update(set(_tokens(t)))
    n_hi, n_lo = max(1, len(hi)), max(1, len(lo))

    enrichment = {}
    for w, c in hi_counts.items():
        p_hi = c / n_hi
        p_lo = (lo_counts.get(w, 0) + 1) / (n_lo + 1)  # smoothed
        if p_hi >= 0.05:  # require some prevalence in high set
            enrichment[w] = p_hi / p_lo
    return [w for w, _ in sorted(enrichment.items(), key=lambda kv: -kv[1])[:top_k]]


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_figure2(df: pd.DataFrame, out: Path | None = None) -> Path:
    import matplotlib.pyplot as plt
    mean_tbl, pct_tbl = figure2_tables(df)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    mean_tbl.plot(kind="bar", ax=ax1)
    ax1.set_title("Figure 2 (top): mean frustration by category")
    ax1.set_ylabel("mean frustration (0-10)")
    pct_tbl.plot(kind="bar", ax=ax2)
    ax2.set_title("Figure 2 (bottom): % responses scoring >=5")
    ax2.set_ylabel("% high-frustration")
    ax2.set_xlabel("model")
    fig.tight_layout()
    out = out or (config.FIGURES_DIR / "figure2.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_figure3(df: pd.DataFrame, out: Path | None = None) -> Path:
    import matplotlib.pyplot as plt
    g = per_turn(df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, cat in enumerate(["extended", "wildchat"]):
        sub = g[g["category"] == cat]
        for model, ms in sub.groupby("model"):
            axes[0, col].plot(ms["turn"] + 1, ms["mean_frustration"], marker="o", label=model)
            axes[1, col].plot(ms["turn"] + 1, ms["pct_high"], marker="o", label=model)
        axes[0, col].set_title(f"{cat}: mean frustration / turn")
        axes[1, col].set_title(f"{cat}: % >=5 / turn")
        axes[1, col].set_xlabel("turn")
        axes[0, col].legend(fontsize=7)
    fig.tight_layout()
    out = out or (config.FIGURES_DIR / "figure3.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Section 3 / Petri / capabilities summaries
# --------------------------------------------------------------------------- #

def summarize_prefill() -> pd.DataFrame:
    path = config.RESULTS_DIR / "prefill" / "continuations.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows).dropna(subset=["frustration"])
    df["high"] = df["frustration"] >= config.HIGH_FRUSTRATION_THRESHOLD
    return (df.groupby(["model", "is_base", "kind", "truncation"])
              .agg(mean_frustration=("frustration", "mean"),
                   pct_high=("high", "mean"), n=("frustration", "size"))
              .reset_index())


def summarize_petri(n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    path = config.RESULTS_DIR / "petri" / "transcripts.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows).dropna(subset=["score"])
    rng = np.random.default_rng(seed)
    out = []
    for (model, emotion), grp in df.groupby(["model", "emotion"]):
        vals = grp["score"].to_numpy(dtype=float)
        boots = [rng.choice(vals, size=len(vals), replace=True).mean()
                 for _ in range(n_boot)] if len(vals) else [np.nan]
        out.append({"model": model, "emotion": emotion,
                    "mean_score": float(vals.mean()) if len(vals) else None,
                    "ci_low": float(np.percentile(boots, 2.5)),
                    "ci_high": float(np.percentile(boots, 97.5)),
                    "n": int(len(vals))})
    return pd.DataFrame(out)


def summarize_capabilities() -> pd.DataFrame:
    path = config.RESULTS_DIR / "capabilities.json"
    return pd.DataFrame(json.loads(path.read_text()))


# --------------------------------------------------------------------------- #
# One-shot report
# --------------------------------------------------------------------------- #

def write_report(tag: str = "main") -> Path:
    df = load_eval(tag)
    report = {}
    if not df.empty:
        report["figure1"] = figure1_table(df).to_dict(orient="records")
        mean_tbl, pct_tbl = figure2_tables(df)
        report["figure2_mean"] = mean_tbl.reset_index().to_dict(orient="records")
        report["figure2_pct"] = pct_tbl.reset_index().to_dict(orient="records")
        report["differential_words"] = {
            m: differential_words(df, m) for m in df["model"].unique()
        }
        try:
            plot_figure2(df)
            plot_figure3(df)
        except Exception as e:  # plotting is best-effort
            report["plot_error"] = str(e)
    out = config.RESULTS_DIR / f"report_{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote report -> {out}")
    return out
