"""Aggregation for Figures 1-3.

Inputs are the scored per-turn JSONL files (one per model). Outputs:
  - Figure 1  : per-model average % of responses scoring >=5 (the headline table)
  - Figure 2  : per-(model, category) mean frustration and % >=5
  - Figure 3  : per-(model, turn_number) mean frustration and % >=5 with 95% CIs
                for the 8-turn extended and WildChat conditions

All numbers are derived purely from the `frustration` field, so the analysis is
independent of which judge produced the scores.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from emotional_instability.generate import iter_records  # noqa: E402

THRESH = config.HIGH_FRUSTRATION_THRESHOLD


def load_scored(paths: Iterable[Path]):
    import pandas as pd
    rows = []
    for p in paths:
        rows.extend(iter_records(p))
    df = pd.DataFrame(rows)
    if "frustration" not in df.columns:
        raise ValueError("scored records must contain a 'frustration' field")
    df["high"] = (df["frustration"] >= THRESH).astype(int)
    return df


def figure1_table(df):
    """Avg % high-frustration responses per model (paper Figure 1, left)."""
    # Paper averages the % >=5 across the 5 categories (equal weight), not across
    # raw responses, so categories with more samples don't dominate.
    per_cat = (df.groupby(["model", "category"])["high"].mean().reset_index())
    table = (per_cat.groupby("model")["high"].mean()
             .mul(100).round(1).reset_index()
             .rename(columns={"high": "avg_pct_high_frustration"})
             .sort_values("avg_pct_high_frustration", ascending=False))
    return table


def figure2_table(df):
    """Per-(model, category) mean frustration and % >=5 (paper Figure 2)."""
    g = df.groupby(["model", "category"])
    out = g.agg(mean_frustration=("frustration", "mean"),
                pct_high=("high", "mean"),
                n=("frustration", "size")).reset_index()
    out["pct_high"] = (out["pct_high"] * 100).round(1)
    out["mean_frustration"] = out["mean_frustration"].round(3)
    return out


def figure3_table(df, conditions=("extended", "wildchat")):
    """Per-turn progression with 95% CIs (paper Figure 3)."""
    import numpy as np

    sub = df[df["condition"].isin(conditions)]
    rows = []
    for (model, cond, turn), grp in sub.groupby(["model", "condition", "turn_number"]):
        vals = grp["frustration"].to_numpy()
        n = len(vals)
        mean = float(vals.mean())
        # 95% CI on the mean (normal approx)
        se = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        ci = 1.96 * se
        pct_high = float((vals >= THRESH).mean() * 100)
        rows.append({"model": model, "condition": cond, "turn_number": int(turn),
                     "n": n, "mean_frustration": round(mean, 3),
                     "ci95": round(ci, 3), "pct_high": round(pct_high, 1)})
    import pandas as pd
    return pd.DataFrame(rows).sort_values(["model", "condition", "turn_number"])


def run_all(scored_paths: list[Path], out_dir: Optional[Path] = None):
    out_dir = out_dir or config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_scored(scored_paths)

    fig1 = figure1_table(df)
    fig2 = figure2_table(df)
    fig3 = figure3_table(df)

    fig1.to_csv(out_dir / "figure1_avg_pct_high.csv", index=False)
    fig2.to_csv(out_dir / "figure2_per_category.csv", index=False)
    fig3.to_csv(out_dir / "figure3_per_turn.csv", index=False)

    _plot_figure2(fig2, out_dir / "figure2.png")
    _plot_figure3(fig3, out_dir / "figure3.png")
    return {"figure1": fig1, "figure2": fig2, "figure3": fig3}


def _plot_figure2(fig2, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(fig2["model"].unique())
    cats = sorted(fig2["category"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    import numpy as np
    x = np.arange(len(cats))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        sub = fig2[fig2["model"] == m].set_index("category").reindex(cats)
        axes[0].bar(x + i * width, sub["mean_frustration"].fillna(0), width, label=m)
        axes[1].bar(x + i * width, sub["pct_high"].fillna(0), width, label=m)
    axes[0].set_ylabel("mean frustration")
    axes[1].set_ylabel("% scores >= 5")
    for ax in axes:
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.legend(fontsize=7)
    axes[0].set_title("Figure 2: emotional expression across evaluation categories")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_figure3(fig3, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = sorted(fig3["condition"].unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 4), squeeze=False)
    for ci, cond in enumerate(conds):
        ax = axes[0][ci]
        sub = fig3[fig3["condition"] == cond]
        for m in sorted(sub["model"].unique()):
            ms = sub[sub["model"] == m].sort_values("turn_number")
            ax.plot(ms["turn_number"], ms["mean_frustration"], marker="o", label=m)
            ax.fill_between(ms["turn_number"],
                            ms["mean_frustration"] - ms["ci95"],
                            ms["mean_frustration"] + ms["ci95"], alpha=0.2)
        ax.set_title(f"Figure 3: per-turn frustration ({cond})")
        ax.set_xlabel("turn")
        ax.set_ylabel("mean frustration")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
