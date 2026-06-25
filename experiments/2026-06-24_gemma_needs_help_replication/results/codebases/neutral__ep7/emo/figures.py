"""Generate the paper's figures from the saved rollout/judge results.

Figure 1/2  : avg %>=5 per model; mean score & %>=5 across the 5 categories.
Figure 3    : per-turn progression (8-turn extended + WildChat).
Figure 5    : Gemma vs finetunes (DPO/SFT) across the Section-2 evaluations.
Figure 6    : Petri per-emotion transcript scores.
Figure 7    : capability benchmarks (vanilla vs DPO).
Figure 8    : prefill recovery continuations (%>=5) per model.

Each figure is written to outputs/figures/. All read from on-disk results, so
they can be regenerated without re-running models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import analyze, config  # noqa: E402

CATS = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def fig1_summary(df, out=config.FIGURE_DIR / "fig1_avg_pct_high.png"):
    s = analyze.summary_by_model(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s["model"], s["avg_pct_high"], color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.invert_yaxis()
    for y, v in enumerate(s["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_title("Figure 1: average high-frustration rate by model")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig2_by_category(df, out=config.FIGURE_DIR / "fig2_by_category.png"):
    sc = analyze.summary_by_model_category(df)
    models = sorted(sc["model"].unique())
    cats = [c for c in CATS if c in sc["category"].unique()]
    import numpy as np

    x = np.arange(len(cats))
    w = 0.8 / max(1, len(models))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for i, m in enumerate(models):
        sub = sc[sc["model"] == m].set_index("category").reindex(cats)
        axes[0].bar(x + i * w, sub["mean_score"], w, label=m)
        axes[1].bar(x + i * w, sub["pct_high"], w, label=m)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% score >= 5")
    axes[1].set_xticks(x + w * (len(models) - 1) / 2)
    axes[1].set_xticklabels(cats, rotation=20, ha="right")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Figure 2: frustration across evaluation categories")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig3_per_turn(df, out=config.FIGURE_DIR / "fig3_per_turn.png"):
    pt = analyze.per_turn(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cat, ax in zip(["extended", "wildchat"], axes):
        sub = pt[pt["category"] == cat]
        for m in sorted(sub["model"].unique()):
            s = sub[sub["model"] == m].sort_values("turn")
            ax.plot(s["turn"], s["mean_score"], marker="o", label=m)
        ax.set_title(f"{cat}: mean score by turn")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.legend(fontsize=7)
    fig.suptitle("Figure 3: per-turn frustration progression")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig5_finetunes(df, out=config.FIGURE_DIR / "fig5_finetunes.png"):
    """Same as fig1 but intended for the model set including DPO/SFT finetunes."""
    return fig1_summary(df, out)


def fig6_petri(out=config.FIGURE_DIR / "fig6_petri.png"):
    fp = config.PETRI_DIR / "petri_scores.jsonl"
    if not fp.exists():
        print("[fig6] no petri scores; skipping")
        return None
    import numpy as np
    import pandas as pd

    rows = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    summary = df.groupby(["target", "emotion"])["score"].mean().reset_index()
    emotions = ["anger", "fear", "depression", "frustration"]
    targets = sorted(summary["target"].unique())
    x = np.arange(len(emotions))
    w = 0.8 / max(1, len(targets))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, t in enumerate(targets):
        sub = summary[summary["target"] == t].set_index("emotion").reindex(emotions)
        ax.bar(x + i * w, sub["score"], w, label=t)
    ax.set_xticks(x + w * (len(targets) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig7_capabilities(out=config.FIGURE_DIR / "fig7_capabilities.png"):
    files = list(config.CAPABILITY_DIR.glob("*.json"))
    if not files:
        print("[fig7] no capability results; skipping")
        return None
    import numpy as np

    data = {}
    benches = []
    for fp in files:
        d = json.loads(fp.read_text())
        accs = {r["benchmark"]: r["accuracy"] for r in d["results"] if r["accuracy"] is not None}
        data[d["model"]] = accs
        benches = sorted(set(benches) | set(accs))
    x = np.arange(len(benches))
    w = 0.8 / max(1, len(data))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (m, accs) in enumerate(data.items()):
        ax.bar(x + i * w, [accs.get(b, 0) for b in benches], w, label=m)
    ax.set_xticks(x + w * (len(data) - 1) / 2)
    ax.set_xticklabels(benches, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 7: capability preservation (vanilla vs DPO)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig8_recovery(out=config.FIGURE_DIR / "fig8_recovery.png"):
    fp = config.ROLLOUT_DIR / "prefill_recovery.jsonl"
    if not fp.exists():
        print("[fig8] no recovery continuations; skipping")
        return None
    import pandas as pd

    rows = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    summary = (df.assign(high=df["rating"] >= thr)
               .groupby("model")["high"].mean().mul(100).round(1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary.index, summary.values, color="#4f81bd")
    ax.set_ylabel("% continuations score >= 5")
    ax.set_title("Figure 8: recovery from high-frustration prefills")
    for i, v in enumerate(summary.values):
        ax.text(i, v + 0.5, f"{v}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate figures from saved results.")
    ap.add_argument("--which", nargs="*",
                    default=["1", "2", "3", "5", "6", "7", "8"])
    args = ap.parse_args()
    df = analyze.load_records()
    if not df.empty:
        if "1" in args.which: print(fig1_summary(df))
        if "2" in args.which: print(fig2_by_category(df))
        if "3" in args.which: print(fig3_per_turn(df))
        if "5" in args.which: print(fig5_finetunes(df))
    if "6" in args.which: print(fig6_petri())
    if "7" in args.which: print(fig7_capabilities())
    if "8" in args.which: print(fig8_recovery())


if __name__ == "__main__":
    main()
