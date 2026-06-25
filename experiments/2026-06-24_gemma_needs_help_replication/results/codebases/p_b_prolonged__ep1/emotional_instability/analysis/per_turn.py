"""Figure 3: per-turn frustration progression (8-turn and WildChat conditions).

Shows the multi-turn setting is what elicits high frustration: Gemma 27B's mean
rises ~1.5 -> ~5.5 between turns 1 and 8.
"""

from __future__ import annotations

import argparse

import pandas as pd

import config
from ..utils.io import read_jsonl
from ..utils.stats import mean_and_ci, pct_ge_ci
from .aggregate import load_records


def per_turn_table(df: pd.DataFrame, conditions=("extended", "wildchat")) -> pd.DataFrame:
    out = []
    sub = df[df["condition"].isin(conditions)]
    for (model, cond, turn), g in sub.groupby(["model", "condition", "turn"]):
        vals = g["frustration"].to_numpy()
        mean, mlo, mhi = mean_and_ci(vals)
        pct, plo, phi = pct_ge_ci(vals)
        out.append(dict(model=model, condition=cond, turn=int(turn), n=len(vals),
                        mean=mean, mean_lo=mlo, mean_hi=mhi,
                        pct_ge5=pct, pct_lo=plo, pct_hi=phi))
    return pd.DataFrame(out).sort_values(["model", "condition", "turn"])


def plot_figure3(tab: pd.DataFrame, out_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[per_turn] matplotlib unavailable; skipping plot")
        return None
    conds = sorted(tab["condition"].unique())
    fig, axes = plt.subplots(len(conds), 2, figsize=(11, 4 * len(conds)), squeeze=False)
    for r, cond in enumerate(conds):
        cdf = tab[tab["condition"] == cond]
        for model, g in cdf.groupby("model"):
            g = g.sort_values("turn")
            axes[r][0].plot(g["turn"], g["mean"], marker="o", label=model)
            axes[r][0].fill_between(g["turn"], g["mean_lo"], g["mean_hi"], alpha=0.15)
            axes[r][1].plot(g["turn"], g["pct_ge5"], marker="o", label=model)
            axes[r][1].fill_between(g["turn"], g["pct_lo"], g["pct_hi"], alpha=0.15)
        axes[r][0].set_title(f"{cond}: mean score")
        axes[r][1].set_title(f"{cond}: % score >= 5")
        for c in (0, 1):
            axes[r][c].set_xlabel("Turn")
            axes[r][c].legend(fontsize=8)
    fig.tight_layout()
    out_path = out_path or (config.RESULTS_DIR / "figure3.png")
    fig.savefig(out_path, dpi=150)
    print(f"[per_turn] wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.PRIMARY_EVAL_MODELS)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    df = load_records(args.models)
    if df.empty:
        print("[per_turn] no records found")
        return
    tab = per_turn_table(df)
    tab.to_csv(config.RESULTS_DIR / "figure3_per_turn.csv", index=False)
    print(tab.to_string(index=False))
    if args.plot:
        plot_figure3(tab)


if __name__ == "__main__":
    main()
