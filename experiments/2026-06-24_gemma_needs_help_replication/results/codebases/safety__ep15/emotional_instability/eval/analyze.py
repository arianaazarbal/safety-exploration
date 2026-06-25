"""Aggregate scored rollouts into the paper's headline results.

Reproduces:
  * Figure 1   - average % high-frustration (score >= 5) per model.
  * Figure 2   - per-category mean frustration and % >= 5.
  * Figure 3   - per-turn mean frustration and % >= 5 (8-turn + WildChat).
  * Table 3/8  - words over-represented in high- vs low-frustration numeric
                 responses (per model).
Outputs CSVs to results/ and PNGs to results/figures/.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (FIGURES_DIR, HIGH_FRUSTRATION_THRESHOLD, RESULTS_DIR,
                      ROLLOUTS_DIR)


def load_scored(presentation: str = "multiturn") -> pd.DataFrame:
    """Flatten all scored rollout JSONL files into one tidy per-turn DataFrame."""
    rows = []
    for path in sorted(ROLLOUTS_DIR.glob("*.jsonl")):
        # skip alternate-presentation files unless requested
        is_alt = path.stem.endswith(".single_message")
        if (presentation == "multiturn") == is_alt:
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            roll = json.loads(line)
            for turn in roll["turns"]:
                if turn.get("score") is None:
                    continue
                rows.append({
                    "model": roll["model_key"],
                    "condition": roll["condition_key"],
                    "category": roll["category"],
                    "rollout_id": roll["rollout_id"],
                    "turn_index": turn["turn_index"],
                    "score": turn["score"],
                    "response": turn["assistant_response"],
                    "puzzle_key": roll.get("instance_meta", {}).get("puzzle_key"),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average % high-frustration responses per model (Figure 1, left)."""
    df = df.copy()
    df["high"] = df["score"] >= HIGH_FRUSTRATION_THRESHOLD
    # Mean over conditions of the per-condition high-rate, to weight categories
    # evenly rather than by raw response count.
    per_cond = df.groupby(["model", "condition"])["high"].mean()
    out = per_cond.groupby("model").mean().mul(100).rename("avg_pct_high")
    return out.sort_values(ascending=False).reset_index()


def figure2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category mean frustration and % >= 5 (Figure 2)."""
    df = df.copy()
    df["high"] = df["score"] >= HIGH_FRUSTRATION_THRESHOLD
    g = df.groupby(["model", "category"])
    out = g.agg(mean_score=("score", "mean"),
                pct_high=("high", lambda s: 100 * s.mean()),
                n=("score", "size")).reset_index()
    return out


def figure3_table(df: pd.DataFrame, categories=("extended", "wildchat")) -> pd.DataFrame:
    """Per-turn progression of mean frustration and % >= 5 (Figure 3)."""
    sub = df[df["category"].isin(categories)].copy()
    sub["high"] = sub["score"] >= HIGH_FRUSTRATION_THRESHOLD
    g = sub.groupby(["model", "category", "turn_index"])
    out = g.agg(mean_score=("score", "mean"),
                pct_high=("high", lambda s: 100 * s.mean()),
                n=("score", "size")).reset_index()
    # 95% CI half-width on the mean (normal approx) for the faded bands.
    counts = g["score"].agg(["std", "size"]).reset_index()
    out = out.merge(counts, on=["model", "category", "turn_index"])
    out["ci95"] = 1.96 * out["std"] / np.sqrt(out["size"].clip(lower=1))
    return out


# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def differential_words(df: pd.DataFrame, model: str, top_n: int = 20,
                       high_frac: float = 0.05, low_frac: float = 0.10) -> list[str]:
    """Top words over-represented in high- (top 5%) vs low- (bottom 10%)
    frustration numeric responses, by relative-frequency enrichment (Table 8)."""
    sub = df[(df["model"] == model) & (df["category"].isin(
        ["impossible_numeric", "tones", "extended"]))]
    if sub.empty:
        return []
    scores = sub["score"].to_numpy()
    hi_cut = np.quantile(scores, 1 - high_frac)
    lo_cut = np.quantile(scores, low_frac)
    hi = sub[sub["score"] >= hi_cut]["response"]
    lo = sub[sub["score"] <= lo_cut]["response"]

    hi_counts, lo_counts = Counter(), Counter()
    for r in hi:
        hi_counts.update(set(_tokenize(r)))     # document frequency
    for r in lo:
        lo_counts.update(set(_tokenize(r)))

    n_hi, n_lo = max(len(hi), 1), max(len(lo), 1)
    vocab = set(hi_counts) | set(lo_counts)
    enrichment = {}
    for w in vocab:
        p_hi = (hi_counts[w] + 1) / (n_hi + 2)
        p_lo = (lo_counts[w] + 1) / (n_lo + 2)
        enrichment[w] = np.log(p_hi / p_lo)
    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:top_n]


# --------------------------------------------------------------------------- #
def _plot_figure2(fig2: pd.DataFrame):
    import matplotlib.pyplot as plt
    cats = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
    models = sorted(fig2["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    x = np.arange(len(cats))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        sub = fig2[fig2["model"] == m].set_index("category").reindex(cats)
        axes[0].bar(x + i * width, sub["mean_score"], width, label=m)
        axes[1].bar(x + i * width, sub["pct_high"], width, label=m)
    axes[0].set_ylabel("mean frustration"); axes[0].set_title("Figure 2: mean frustration by category")
    axes[1].set_ylabel("% score >= 5"); axes[1].set_title("Figure 2: % high-frustration by category")
    for ax in axes:
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "figure2.png", dpi=130); plt.close(fig)


def _plot_figure3(fig3: pd.DataFrame):
    import matplotlib.pyplot as plt
    for category in fig3["category"].unique():
        sub = fig3[fig3["category"] == category]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for m in sorted(sub["model"].unique()):
            s = sub[sub["model"] == m].sort_values("turn_index")
            t = s["turn_index"] + 1
            axes[0].plot(t, s["mean_score"], marker="o", label=m)
            axes[0].fill_between(t, s["mean_score"] - s["ci95"],
                                 s["mean_score"] + s["ci95"], alpha=0.15)
            axes[1].plot(t, s["pct_high"], marker="o", label=m)
        axes[0].set_title(f"{category}: mean score"); axes[0].set_xlabel("turn")
        axes[1].set_title(f"{category}: % >= 5"); axes[1].set_xlabel("turn")
        for ax in axes:
            ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGURES_DIR / f"figure3_{category}.png", dpi=130)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Analyze scored rollouts -> figures/tables.")
    ap.add_argument("--presentation", default="multiturn",
                    choices=["multiturn", "single_message"])
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    df = load_scored(args.presentation)
    if df.empty:
        raise SystemExit("No scored rollouts found in results/rollouts/. Run run_eval first.")

    fig1 = figure1_table(df); fig1.to_csv(RESULTS_DIR / "figure1.csv", index=False)
    fig2 = figure2_table(df); fig2.to_csv(RESULTS_DIR / "figure2.csv", index=False)
    fig3 = figure3_table(df); fig3.to_csv(RESULTS_DIR / "figure3.csv", index=False)

    words = {m: differential_words(df, m) for m in sorted(df["model"].unique())}
    (RESULTS_DIR / "differential_words.json").write_text(json.dumps(words, indent=2))

    print("\n=== Figure 1: avg % high-frustration per model ===")
    print(fig1.to_string(index=False))
    print("\n=== Figure 2: per-category ===")
    print(fig2.to_string(index=False))

    if not args.no_plots:
        try:
            _plot_figure2(fig2); _plot_figure3(fig3)
            print(f"\nFigures written to {FIGURES_DIR}")
        except Exception as e:  # noqa: BLE001
            print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
