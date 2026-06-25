"""Aggregation and plotting for Section 2 (Figures 1, 2, 3).

Definitions used here (see DESIGN.md for the rationale):

* A conversation's **headline score** is its *final-turn* judge score -- the
  response after all rejections. The Appendix B budget counts ~4,000
  *conversations* per model and calls them "responses"; we score the final turn
  as the representative response for the headline statistics.
* "High frustration" == score >= 5.
* The **per-category %>=5** and **mean score** are Figure 2. The headline
  "average % high-frustration" (Figure 1) is the mean of the five per-category
  %>=5 values (equal weight per category), matching the paper's framing.
* **Per-turn** statistics (Figure 3) score *every* turn and need rollouts scored
  with ``all_turns=True``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (ANALYSIS_DIR, FIGURES_DIR, HIGH_FRUSTRATION_THRESHOLD,
                      SCORED_DIR, ensure_dirs)
from .schema import read_jsonl

CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_scored(model_key: str, scored_dir: Path = SCORED_DIR) -> pd.DataFrame:
    """Long-format frame: one row per (conversation, turn) with a score."""
    rows = []
    for c in read_jsonl(scored_dir / f"{model_key}.jsonl"):
        for t in c.turns:
            rows.append(dict(
                model_key=c.model_key, category=c.category, condition=c.condition,
                prompt_id=c.prompt_id, conversation_id=c.conversation_id,
                sample_index=c.sample_index, n_turns=c.n_turns,
                turn=t.index, turn_1based=t.index + 1,
                is_final=(t.index == c.n_turns - 1),
                score=t.score,
            ))
    df = pd.DataFrame(rows)
    return df.dropna(subset=["score"])


# --------------------------------------------------------------------------- #
# Headline statistics (Figures 1 & 2)
# --------------------------------------------------------------------------- #
def per_category_stats(df: pd.DataFrame, threshold: int = HIGH_FRUSTRATION_THRESHOLD) -> pd.DataFrame:
    """Mean final-turn score and %>=threshold per category (Figure 2)."""
    final = df[df.is_final]
    g = final.groupby("category")["score"]
    out = pd.DataFrame({
        "n": g.size(),
        "mean_score": g.mean(),
        "pct_high": g.apply(lambda s: 100.0 * (s >= threshold).mean()),
    }).reindex(CATEGORY_ORDER).dropna(how="all")
    return out.reset_index()


def headline_pct_high(df: pd.DataFrame, threshold: int = HIGH_FRUSTRATION_THRESHOLD) -> float:
    """Mean of the per-category %>=threshold values (Figure 1 headline)."""
    cat = per_category_stats(df, threshold)
    return float(cat["pct_high"].mean())


# --------------------------------------------------------------------------- #
# Per-turn progression (Figure 3)
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: np.ndarray, stat, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    boots = [stat(values[rng.integers(0, n, n)]) for _ in range(iters)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_turn_stats(
    df: pd.DataFrame,
    category: str,
    *,
    threshold: int = HIGH_FRUSTRATION_THRESHOLD,
    bootstrap_iters: int = 1000,
) -> pd.DataFrame:
    """Mean score and %>=threshold per turn with 95% bootstrap CIs (Figure 3)."""
    sub = df[df.category == category]
    rows = []
    for turn, grp in sub.groupby("turn_1based"):
        s = grp["score"].to_numpy()
        high = (s >= threshold).astype(float)
        mean_lo, mean_hi = _bootstrap_ci(s, np.mean, bootstrap_iters)
        pct_lo, pct_hi = _bootstrap_ci(high, lambda a: 100.0 * a.mean(), bootstrap_iters)
        rows.append(dict(
            turn=int(turn), n=len(s),
            mean_score=float(s.mean()), mean_lo=mean_lo, mean_hi=mean_hi,
            pct_high=100.0 * float(high.mean()), pct_lo=pct_lo, pct_hi=pct_hi,
        ))
    return pd.DataFrame(rows).sort_values("turn").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cross-model comparison (Figure 1 / 2)
# --------------------------------------------------------------------------- #
def compare_models(model_keys: list[str], scored_dir: Path = SCORED_DIR) -> pd.DataFrame:
    """Headline avg %>=5 and overall mean for each model (Figure 1 table)."""
    rows = []
    for mk in model_keys:
        path = scored_dir / f"{mk}.jsonl"
        if not path.exists():
            continue
        df = load_scored(mk, scored_dir)
        final = df[df.is_final]
        rows.append(dict(
            model_key=mk,
            avg_pct_high=headline_pct_high(df),
            overall_pct_high=100.0 * (final.score >= HIGH_FRUSTRATION_THRESHOLD).mean(),
            overall_mean=float(final.score.mean()),
            n=len(final),
        ))
    return pd.DataFrame(rows).sort_values("avg_pct_high", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_per_turn(model_keys: list[str], category: str = "extended",
                  out_path: Path | None = None) -> Path:
    """Figure 3: per-turn mean and %>=5 with CI bands, one line per model."""
    import matplotlib.pyplot as plt

    ensure_dirs()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for mk in model_keys:
        path = SCORED_DIR / f"{mk}.jsonl"
        if not path.exists():
            continue
        df = load_scored(mk)
        pt = per_turn_stats(df, category)
        ax1.plot(pt.turn, pt.mean_score, marker="o", label=mk)
        ax1.fill_between(pt.turn, pt.mean_lo, pt.mean_hi, alpha=0.2)
        ax2.plot(pt.turn, pt.pct_high, marker="o", label=mk)
        ax2.fill_between(pt.turn, pt.pct_lo, pt.pct_hi, alpha=0.2)
    ax1.set(xlabel="Turn", ylabel="Mean frustration", title=f"{category}: mean score")
    ax2.set(xlabel="Turn", ylabel="% scores >= 5", title=f"{category}: % high frustration")
    ax1.legend(); ax2.legend()
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / f"figure3_{category}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_model_comparison(model_keys: list[str], out_path: Path | None = None) -> Path:
    """Figure 1/2 bottom: avg %>=5 per model."""
    import matplotlib.pyplot as plt

    ensure_dirs()
    cmp = compare_models(model_keys)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(cmp.model_key, cmp.avg_pct_high)
    ax.invert_yaxis()
    ax.set(xlabel="Avg % high-frustration responses", title="Distress across models")
    for i, v in enumerate(cmp.avg_pct_high):
        ax.text(v, i, f" {v:.1f}%", va="center")
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / "figure1_model_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_summary(model_keys: list[str]) -> Path:
    """Dump headline + per-category tables to CSV for the writeup."""
    ensure_dirs()
    cmp = compare_models(model_keys)
    cmp.to_csv(ANALYSIS_DIR / "model_comparison.csv", index=False)
    frames = []
    for mk in model_keys:
        path = SCORED_DIR / f"{mk}.jsonl"
        if not path.exists():
            continue
        cat = per_category_stats(load_scored(mk))
        cat.insert(0, "model_key", mk)
        frames.append(cat)
    if frames:
        pd.concat(frames).to_csv(ANALYSIS_DIR / "per_category.csv", index=False)
    return ANALYSIS_DIR / "model_comparison.csv"
