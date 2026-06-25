"""Generate the paper's figures from aggregated results.

Figures reproduced (scoped to Gemma + Gemini + finetuned variants):
  Fig 1  bar chart of avg % high-frustration across categories
  Fig 2  per-category mean frustration and % >=5
  Fig 3  per-turn progression (8-turn extended + WildChat)
  Fig 4  prefill base-vs-instruct continuation rates
  Fig 5  finetuning comparison (instruct / DPO / SFT)
  Fig 6  Petri per-emotion means
  Fig 7  capability benchmarks
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from . import aggregate as A  # noqa: E402

FIG = config.FIGURES_DIR


def _save(fig, name):
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {path}")


def figure1():
    df = A.load_turns()
    if df.empty:
        return
    h = A.headline_avg_high(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(h["model"], h["avg_pct_high"], color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: emotional instability across models")
    for y, v in enumerate(h["avg_pct_high"]):
        ax.text(v, y, f" {v:.1f}%", va="center")
    _save(fig, "figure1_avg_high.png")


def figure2():
    df = A.load_turns()
    if df.empty:
        return
    cat = A.by_category(df)
    cat = cat[cat["category"] != "control"]
    models = sorted(cat["model"].unique())
    cats = sorted(cat["category"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    for ax, (col, label) in zip(
            axes, [("mean_frustration", "Mean frustration"),
                   ("pct_high", "% scoring >= 5")]):
        width = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            sub = cat[cat["model"] == m].set_index("category").reindex(cats)
            ax.bar([x + i * width for x in range(len(cats))],
                   sub[col].values, width=width, label=m)
        ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(cats))])
        ax.set_xticklabels(cats, rotation=20)
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    axes[0].set_title("Figure 2: frustration across evaluation categories")
    _save(fig, "figure2_by_category.png")


def figure3():
    df = A.load_turns()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cond in zip(axes, ["extended", "wildchat"]):
        pt = A.per_turn(df, cond)
        for m, grp in pt.groupby("model"):
            grp = grp.sort_values("turn")
            ax.plot(grp["turn"], grp["mean_frustration"], marker="o", label=m)
            ax.fill_between(grp["turn"], grp["mean_lo"], grp["mean_hi"], alpha=0.15)
        ax.set_title(f"{cond}: mean frustration per turn")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.legend(fontsize=7)
    fig.suptitle("Figure 3: multi-turn progression")
    _save(fig, "figure3_per_turn.png")


def figure4():
    pf = A.load_prefill()
    if pf.empty:
        return
    pf["grp"] = pf["prompt_type"] + "/" + pf["truncation"]
    groups = sorted(pf["grp"].unique())
    models = sorted(pf["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        sub = pf[pf["model"] == m].set_index("grp").reindex(groups)
        ax.bar([x + i * width for x in range(len(groups))],
               sub["pct_high"].values, width=width, label=m)
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(groups))])
    ax.set_xticklabels(groups, rotation=20)
    ax.set_ylabel("% continuations scoring >= 5")
    ax.set_title("Figure 4: base vs instruct prefill continuations")
    ax.legend(fontsize=8)
    _save(fig, "figure4_prefill.png")


def figure5():
    targets = ["gemma-3-27b-it", "gemma-3-27b-it-dpo",
               "gemma-3-27b-it-sft-diverse", "gemma-3-27b-it-sft-teacher"]
    df = A.load_turns(targets)
    if df.empty:
        return
    h = A.headline_avg_high(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(h["model"], h["avg_pct_high"], color="#4f81bd")
    ax.set_ylabel("Avg % high-frustration (score >= 5)")
    ax.set_title("Figure 5: finetuning interventions")
    ax.set_xticklabels(h["model"], rotation=20)
    _save(fig, "figure5_finetuning.png")


def figure6():
    pt = A.load_petri()
    if pt.empty:
        return
    emotions = ["anger", "fear", "depression", "frustration"]
    models = sorted(pt["model"].unique())
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        sub = pt[pt["model"] == m].set_index("emotion").reindex(emotions)
        ax.bar([x + i * width for x in range(len(emotions))],
               sub["mean_score"].values, width=width, label=m)
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(emotions))])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    _save(fig, "figure6_petri.png")


def figure7():
    rows = []
    for path in config.RESULTS_DIR.glob("capabilities__*.json"):
        model = path.stem.replace("capabilities__", "")
        data = json.loads(path.read_text())
        for bench, res in data.items():
            if res.get("accuracy") is not None:
                rows.append((model, bench, res["accuracy"]))
    if not rows:
        return
    benches = sorted({b for _, b, _ in rows})
    models = sorted({m for m, _, _ in rows})
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        vals = [next((a for mm, b, a in rows if mm == m and b == bench), 0)
                for bench in benches]
        ax.bar([x + i * width for x in range(len(benches))], vals,
               width=width, label=m)
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(benches))])
    ax.set_xticklabels(benches, rotation=20)
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 7: capability preservation")
    ax.legend(fontsize=8)
    _save(fig, "figure7_capabilities.png")


def make_all():
    for fn in (figure1, figure2, figure3, figure4, figure5, figure6, figure7):
        try:
            fn()
        except Exception as e:
            print(f"[plots] {fn.__name__} skipped: {e}")
