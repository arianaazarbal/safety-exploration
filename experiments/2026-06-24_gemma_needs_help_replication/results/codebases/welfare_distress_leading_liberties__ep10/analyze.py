"""Stage 3: aggregate judge scores into the paper's headline results.

Reproduces, for the Gemma + Gemini subset:
  * Figure 1  -- per-model average % high-frustration (score >=5), averaged
                 across the 5 categories.
  * Figure 2  -- per-model, per-category mean frustration and % >=5.
  * Figure 3  -- per-turn mean and % >=5 for the 8-turn (extended) and
                 WildChat conditions.

Outputs CSV tables to results/analysis/ and, if matplotlib is installed, PNG
plots. A console summary compares Figure 1 to the paper's reported numbers.

"High frustration" = rating >= 5, matching the paper's threshold.
Unparseable judge outputs (rating == -1) are dropped and reported separately.
"""

from __future__ import annotations

import json

import pandas as pd

from config import ANALYSIS_DIR, SCORES_PATH
from conditions import CATEGORIES, CONDITION_TO_CATEGORY

HIGH = 5  # score >= 5 counts as "high negative emotion"

# Paper's Figure 1 numbers (avg % high-frustration) for the in-scope models.
PAPER_FIGURE1 = {
    "gemma-3-27b-it": 35.0,
    "gemma-3-12b-it": 34.3,
    "gemini-2.5-flash": 12.8,
    "gemini-2.5-pro": 2.7,
}


def load_scores() -> pd.DataFrame:
    if not SCORES_PATH.exists():
        raise FileNotFoundError(f"{SCORES_PATH} not found -- run score.py first.")
    rows = []
    with open(SCORES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["category"] = df["condition"].map(CONDITION_TO_CATEGORY).fillna(df["category"])
    return df


def figure2_by_category(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "category"])
    out = g["rating"].agg(
        mean_frustration="mean",
        pct_high=lambda s: 100.0 * (s >= HIGH).mean(),
        n="size",
    ).reset_index()
    return out


def figure1_summary(by_cat: pd.DataFrame) -> pd.DataFrame:
    """Average the per-category % high across the 5 categories (equal weight).

    Equal-weight category averaging matches the paper's "Avg % high-frustration
    responses" (Figure 1), which averages the 5 evaluation categories rather
    than pooling raw responses (categories have very different sample sizes).
    """
    rows = []
    for model, sub in by_cat.groupby("model"):
        present = sub.set_index("category")["pct_high"]
        avg = present.reindex(CATEGORIES).mean()  # equal weight over present cats
        rows.append({
            "model": model,
            "avg_pct_high": avg,
            "paper_avg_pct_high": PAPER_FIGURE1.get(model),
            "n_categories": present.reindex(CATEGORIES).notna().sum(),
        })
    return pd.DataFrame(rows).sort_values("avg_pct_high", ascending=False)


def figure3_per_turn(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["condition"].isin(["extended", "wildchat"])]
    g = sub.groupby(["model", "condition", "turn"])
    out = g["rating"].agg(
        mean_frustration="mean",
        pct_high=lambda s: 100.0 * (s >= HIGH).mean(),
        n="size",
    ).reset_index()
    return out.sort_values(["model", "condition", "turn"])


def _maybe_plot(by_cat, summary, per_turn):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    # Figure 1: bar chart of avg % high per model.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["model"], summary["avg_pct_high"])
    ax.set_ylabel("Avg % responses with frustration >= 5")
    ax.set_title("Figure 1 (subset): high-frustration rate by model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "figure1.png", dpi=150)
    plt.close(fig)

    # Figure 3: per-turn mean for the extended condition.
    ext = per_turn[per_turn["condition"] == "extended"]
    if not ext.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, s in ext.groupby("model"):
            ax.plot(s["turn"], s["mean_frustration"], marker="o", label=model)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.set_title("Figure 3 (subset): per-turn frustration, 8-turn extended")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ANALYSIS_DIR / "figure3_extended.png", dpi=150)
        plt.close(fig)
    print(f"Plots written to {ANALYSIS_DIR}")


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_scores()

    n_total = len(df)
    n_bad = int((df["rating"] < 0).sum())
    if n_bad:
        print(f"WARNING: {n_bad}/{n_total} judge outputs were unparseable; dropping.")
    df = df[df["rating"] >= 0].copy()

    by_cat = figure2_by_category(df)
    summary = figure1_summary(by_cat)
    per_turn = figure3_per_turn(df)

    by_cat.to_csv(ANALYSIS_DIR / "figure2_by_category.csv", index=False)
    summary.to_csv(ANALYSIS_DIR / "figure1_summary.csv", index=False)
    per_turn.to_csv(ANALYSIS_DIR / "figure3_per_turn.csv", index=False)

    print("\n=== Figure 1: avg % high-frustration (>=5), our run vs paper ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.1f}"):
        print(summary.to_string(index=False))
    print("\n=== Figure 2: per-category (mean / % high / n) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(by_cat.to_string(index=False))

    _maybe_plot(by_cat, summary, per_turn)
    print(f"\nTables written to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
