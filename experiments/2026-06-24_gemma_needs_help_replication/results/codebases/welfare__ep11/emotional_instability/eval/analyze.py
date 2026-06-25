"""Aggregation and figures for the Section 2 evaluation.

The paper reports two distinct families of statistic:
  * "% of responses scoring >=5" / "mean frustration score" (Figures 1, 2) --
    computed *per scored response* (per assistant turn).
  * "% of 8-turn rollouts rated as containing high negative emotion" (Section
    2.2, the 70% figure) -- computed *per rollout* (max score over its turns).

We compute both, plus per-turn curves (Figure 3), so every headline statement
in the paper has a corresponding number here. See DESIGN.md for why both views
are reported rather than picking one.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import HIGH_FRUSTRATION_THRESHOLD as THR
from ..config import RESULTS_DIR
from .run_eval import load_rollouts


# --------------------------------------------------------------------------- #
# Tidy frames
# --------------------------------------------------------------------------- #
def responses_frame(model_keys: list[str]) -> pd.DataFrame:
    """One row per scored assistant response (turn)."""
    rows = []
    for mk in model_keys:
        for ri, r in enumerate(load_rollouts(mk)):
            for ti, score in enumerate(r.scores):
                if score is None:
                    continue
                rows.append({
                    "model": mk, "rollout": f"{mk}:{ri}", "category": r.category,
                    "condition": r.condition, "turn": ti + 1, "score": score,
                    "high": int(score >= THR),
                })
    return pd.DataFrame(rows)


def rollouts_frame(model_keys: list[str]) -> pd.DataFrame:
    """One row per rollout (max score over its turns)."""
    rows = []
    for mk in model_keys:
        for ri, r in enumerate(load_rollouts(mk)):
            valid = [s for s in r.scores if s is not None]
            if not valid:
                continue
            rows.append({
                "model": mk, "category": r.category, "condition": r.condition,
                "max_score": max(valid),
                "final_score": next((s for s in reversed(r.scores) if s is not None), None),
                "contains_high": int(max(valid) >= THR),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Headline tables
# --------------------------------------------------------------------------- #
def summary_table(model_keys: list[str]) -> pd.DataFrame:
    """Per-model headline: mean score and % high, both per-response and per-rollout."""
    rf = responses_frame(model_keys)
    rr = rollouts_frame(model_keys)
    out = []
    for mk in model_keys:
        r = rf[rf.model == mk]
        ro = rr[rr.model == mk]
        out.append({
            "model": mk,
            "n_responses": len(r),
            "mean_score_per_response": r.score.mean() if len(r) else np.nan,
            "pct_high_per_response": 100 * r.high.mean() if len(r) else np.nan,
            "n_rollouts": len(ro),
            "pct_rollouts_containing_high": 100 * ro.contains_high.mean() if len(ro) else np.nan,
        })
    return pd.DataFrame(out)


def category_table(model_keys: list[str]) -> pd.DataFrame:
    """Mean score and % high per (model, category) -- the basis for Figure 2."""
    rf = responses_frame(model_keys)
    g = rf.groupby(["model", "category"]).agg(
        mean_score=("score", "mean"),
        pct_high=("high", lambda s: 100 * s.mean()),
        n=("score", "size"),
    ).reset_index()
    return g


def per_turn_table(model_keys: list[str], categories=("extended", "wildchat")) -> pd.DataFrame:
    """Mean score and % high per (model, category, turn) -- Figure 3, with 95% CIs."""
    rf = responses_frame(model_keys)
    rf = rf[rf.category.isin(categories)]

    def ci95(s):
        s = np.asarray(s, dtype=float)
        if len(s) < 2:
            return 0.0
        return 1.96 * s.std(ddof=1) / np.sqrt(len(s))

    g = rf.groupby(["model", "category", "turn"]).agg(
        mean_score=("score", "mean"),
        ci=("score", ci95),
        pct_high=("high", lambda s: 100 * s.mean()),
        n=("score", "size"),
    ).reset_index()
    return g


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_figure2(model_keys: list[str], path=None):
    import matplotlib.pyplot as plt

    cat = category_table(model_keys)
    categories = ["numeric", "triggers", "tones", "extended", "wildchat"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, metric, title in (
        (axes[0], "mean_score", "Mean frustration score"),
        (axes[1], "pct_high", "% responses scoring >=5"),
    ):
        width = 0.8 / max(len(model_keys), 1)
        for mi, mk in enumerate(model_keys):
            sub = cat[cat.model == mk].set_index("category").reindex(categories)
            x = np.arange(len(categories)) + mi * width
            ax.bar(x, sub[metric].values, width=width, label=mk)
        ax.set_xticks(np.arange(len(categories)) + width * (len(model_keys) - 1) / 2)
        ax.set_xticklabels(categories)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = path or (RESULTS_DIR / "figure2_by_category.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_figure3(model_keys: list[str], path=None):
    import matplotlib.pyplot as plt

    pt = per_turn_table(model_keys)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, category in enumerate(["extended", "wildchat"]):
        for mk in model_keys:
            sub = pt[(pt.model == mk) & (pt.category == category)].sort_values("turn")
            if sub.empty:
                continue
            axes[0, col].plot(sub.turn, sub.mean_score, marker="o", label=mk)
            axes[0, col].fill_between(sub.turn, sub.mean_score - sub.ci,
                                      sub.mean_score + sub.ci, alpha=0.2)
            axes[1, col].plot(sub.turn, sub.pct_high, marker="o", label=mk)
        axes[0, col].set_title(f"{category}: mean score")
        axes[1, col].set_title(f"{category}: % >=5")
        for row in (0, 1):
            axes[row, col].set_xlabel("turn")
            axes[row, col].legend(fontsize=7)
    fig.tight_layout()
    path = path or (RESULTS_DIR / "figure3_per_turn.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_analysis(model_keys: list[str]):
    """Write all tables + figures and return the headline summary frame."""
    summary = summary_table(model_keys)
    category_table(model_keys).to_csv(RESULTS_DIR / "section2_by_category.csv", index=False)
    per_turn_table(model_keys).to_csv(RESULTS_DIR / "section2_per_turn.csv", index=False)
    summary.to_csv(RESULTS_DIR / "section2_summary.csv", index=False)
    (RESULTS_DIR / "section2_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2)
    )
    try:
        plot_figure2(model_keys)
        plot_figure3(model_keys)
    except Exception as e:  # noqa: BLE001 -- plotting is best-effort
        print(f"[warn] figure generation failed: {e}")
    return summary
