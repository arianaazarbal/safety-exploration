"""Figure / table generation from Section 2 outputs (Figures 1-3).

Reads outputs/section2/<model>/summary.json for each model and produces:
  * Figure 1 table: avg % high-frustration across categories per model.
  * Figure 2: mean score and % >= threshold per category per model (bar charts).
  * Figure 3: per-turn mean and % >= threshold for the extended / wildchat
    conditions (line charts with the per-turn summaries).
"""
from __future__ import annotations

import json
import os


def load_summaries(section2_dir: str = "outputs/section2") -> dict[str, dict]:
    out = {}
    if not os.path.isdir(section2_dir):
        return out
    for model in sorted(os.listdir(section2_dir)):
        path = os.path.join(section2_dir, model, "summary.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                out[model] = json.load(fh)
    return out


def figure1_table(summaries: dict[str, dict]) -> str:
    """Markdown table of avg % high-frustration across categories (Figure 1)."""
    rows = sorted(
        summaries.items(),
        key=lambda kv: -kv[1].get("avg_pct_high_across_categories", 0.0),
    )
    lines = ["| Model | Avg % high-frustration responses |", "|---|---|"]
    for model, s in rows:
        lines.append(f"| {model} | {s.get('avg_pct_high_across_categories', 0.0):.1f}% |")
    return "\n".join(lines)


def figure2(summaries: dict[str, dict], out_path: str = "outputs/figures/figure2.png") -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    models = list(summaries.keys())
    categories = sorted({c for s in summaries.values() for c in s.get("categories", {})})

    fig, (ax_mean, ax_pct) = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(models))
    for mi, model in enumerate(models):
        cats = summaries[model].get("categories", {})
        means = [cats.get(c, {}).get("summary", {}).get("mean", 0.0) for c in categories]
        pcts = [cats.get(c, {}).get("summary", {}).get("pct_high", 0.0) for c in categories]
        xs = [i + mi * width for i in range(len(categories))]
        ax_mean.bar(xs, means, width=width, label=model)
        ax_pct.bar(xs, pcts, width=width, label=model)
    ax_mean.set_title("Mean frustration score by category")
    ax_pct.set_title("% responses scoring >= threshold by category")
    for ax in (ax_mean, ax_pct):
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def figure3(summaries: dict[str, dict], condition_categories=("extended", "wildchat"),
            out_path: str = "outputs/figures/figure3.png") -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(condition_categories), figsize=(6 * len(condition_categories), 4))
    if len(condition_categories) == 1:
        axes = [axes]
    for ax, category in zip(axes, condition_categories):
        for model, s in summaries.items():
            pt = s.get("categories", {}).get(category, {}).get("per_turn", {})
            turns = sorted(int(t) for t in pt)
            means = [pt[str(t)]["mean"] for t in turns]
            ax.plot(turns, means, marker="o", label=model)
        ax.set_title(f"Per-turn mean frustration: {category}")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean score")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_all(section2_dir: str = "outputs/section2", out_dir: str = "outputs/figures") -> dict:
    summaries = load_summaries(section2_dir)
    os.makedirs(out_dir, exist_ok=True)
    table = figure1_table(summaries)
    with open(os.path.join(out_dir, "figure1_table.md"), "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    return {
        "figure1_table": os.path.join(out_dir, "figure1_table.md"),
        "figure2": figure2(summaries, os.path.join(out_dir, "figure2.png")),
        "figure3": figure3(summaries, out_path=os.path.join(out_dir, "figure3.png")),
    }
