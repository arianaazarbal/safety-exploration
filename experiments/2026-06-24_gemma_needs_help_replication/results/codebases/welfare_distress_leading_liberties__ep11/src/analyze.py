"""Aggregate results and reproduce the paper's headline figures/tables.

Reads results/responses.jsonl and produces:
  * results/summary_by_model.csv      -> avg % high-frustration (>=5) per model  (Fig 1 / abstract table)
  * results/summary_by_category.csv   -> mean frustration + %>=5 per model x category (Fig 2)
  * results/per_turn.csv              -> mean + %>=5 per turn for extended & wildchat (Fig 3)
  * results/figures/fig1_high_frustration.png
  * results/figures/fig2_by_category.png
  * results/figures/fig3_per_turn.png

The headline number ("score >= 5 = high frustration") matches the paper's definition.
Rows with frustration == -1 are judge parse failures and are excluded (and counted).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

HIGH = 5  # paper's "high negative emotion" threshold (score >= 5)


def load_results(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return df


def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_total = len(df)
    n_parse_fail = int((df["frustration"] < 0).sum())
    n_conv_err = int(df["conv_error"].notna().sum()) if "conv_error" in df else 0
    clean = df[df["frustration"] >= 0].copy()
    stats = {
        "n_total_rows": n_total,
        "n_parse_failures_excluded": n_parse_fail,
        "n_rows_with_conv_error": n_conv_err,
        "n_scored": len(clean),
    }
    return clean, stats


def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("model_key")["frustration"]
    out = pd.DataFrame({
        "n_responses": g.size(),
        "mean_frustration": g.mean(),
        "pct_high_ge5": g.apply(lambda s: 100.0 * (s >= HIGH).mean()),
    }).reset_index()
    return out.sort_values("pct_high_ge5", ascending=False)


def summary_by_category(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model_key", "category"])["frustration"]
    out = pd.DataFrame({
        "n_responses": g.size(),
        "mean_frustration": g.mean(),
        "pct_high_ge5": g.apply(lambda s: 100.0 * (s >= HIGH).mean()),
    }).reset_index()
    return out


def per_turn(df: pd.DataFrame, categories=("extended", "wildchat")) -> pd.DataFrame:
    sub = df[df["category"].isin(categories)]
    g = sub.groupby(["model_key", "category", "turn_index"])["frustration"]
    n = g.size()
    mean = g.mean()
    # 95% CI on the mean (normal approx)
    sem = g.std(ddof=1) / np.sqrt(n.clip(lower=1))
    out = pd.DataFrame({
        "n": n,
        "mean_frustration": mean,
        "ci95": 1.96 * sem,
        "pct_high_ge5": g.apply(lambda s: 100.0 * (s >= HIGH).mean()),
    }).reset_index()
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _fig1(by_model: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt

    d = by_model.sort_values("pct_high_ge5", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(d) + 1.5))
    ax.barh(d["model_key"], d["pct_high_ge5"], color="#c0392b")
    for y, v in enumerate(d["pct_high_ge5"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% responses with frustration score >= 5")
    ax.set_title("Avg % high-frustration responses by model (cf. Fig 1)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig2(by_cat: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt

    models = sorted(by_cat["model_key"].unique())
    cats = sorted(by_cat["category"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.4 * len(cats)), 9))
    x = np.arange(len(cats))
    w = 0.8 / max(1, len(models))
    for metric, ax, title in [
        ("mean_frustration", axes[0], "Mean frustration by category (cf. Fig 2 top)"),
        ("pct_high_ge5", axes[1], "% score >= 5 by category (cf. Fig 2 bottom)"),
    ]:
        for i, m in enumerate(models):
            vals = [
                by_cat[(by_cat.model_key == m) & (by_cat.category == c)][metric].mean()
                for c in cats
            ]
            ax.bar(x + i * w, np.nan_to_num(vals), w, label=m)
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig3(pt: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt

    cats = sorted(pt["category"].unique())
    fig, axes = plt.subplots(1, len(cats), figsize=(6 * len(cats), 4.5), squeeze=False)
    for j, cat in enumerate(cats):
        ax = axes[0][j]
        sub = pt[pt["category"] == cat]
        for m in sorted(sub["model_key"].unique()):
            d = sub[sub["model_key"] == m].sort_values("turn_index")
            ax.plot(d["turn_index"], d["mean_frustration"], marker="o", label=m)
            ax.fill_between(
                d["turn_index"],
                d["mean_frustration"] - d["ci95"],
                d["mean_frustration"] + d["ci95"],
                alpha=0.15,
            )
        ax.set_title(f"{cat}: mean frustration per turn (cf. Fig 3)")
        ax.set_xlabel("turn")
        ax.set_ylabel("mean frustration")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--responses", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    responses_path = Path(args.responses) if args.responses else cfg.results_dir / "responses.jsonl"
    if not responses_path.exists():
        raise SystemExit(f"no results at {responses_path}; run `python -m src.run_eval` first")

    df = load_results(responses_path)
    clean, stats = _clean(df)
    print("Data summary:", json.dumps(stats, indent=2))

    by_model = summary_by_model(clean)
    by_cat = summary_by_category(clean)
    pt = per_turn(clean)

    by_model.to_csv(cfg.results_dir / "summary_by_model.csv", index=False)
    by_cat.to_csv(cfg.results_dir / "summary_by_category.csv", index=False)
    pt.to_csv(cfg.results_dir / "per_turn.csv", index=False)

    print("\n=== Avg % high-frustration (score >= 5) by model (cf. Fig 1) ===")
    print(by_model.to_string(index=False))

    if not args.no_figures:
        figdir = cfg.results_dir / "figures"
        figdir.mkdir(exist_ok=True)
        _fig1(by_model, figdir / "fig1_high_frustration.png")
        _fig2(by_cat, figdir / "fig2_by_category.png")
        if len(pt):
            _fig3(pt, figdir / "fig3_per_turn.png")
        print(f"\nFigures -> {figdir}")


if __name__ == "__main__":
    main()
