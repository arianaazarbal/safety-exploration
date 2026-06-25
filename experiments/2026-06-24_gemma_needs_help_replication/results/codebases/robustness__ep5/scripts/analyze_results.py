"""Build the paper's figures/tables from the JSON summaries written by the runs.

Reproduces:
  * Figure 1 / 2 : headline "% high-frustration" bar chart + mean-frustration.
  * Figure 3      : per-turn progression for the 8-turn and WildChat conditions.
  * Table 3 / 8   : differential-word lists per model.

Reads from results/eval/<model>/summary.json (+ rollouts.jsonl) and emits PNGs
and a markdown table to results/figures/.
"""
from __future__ import annotations

# --- PATH SHIM: ensure repo root is importable when run as `python scripts/x.py`
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

from emotional_instability import config_bridge as cfg
from emotional_instability.word_freq import differential_words


def _load_summaries() -> dict:
    out = {}
    eval_dir = cfg.RESULTS_DIR / "eval"
    if not eval_dir.exists():
        return out
    for d in sorted(eval_dir.iterdir()):
        f = d / "summary.json"
        if f.exists():
            out[d.name] = json.loads(f.read_text())
    return out


def figure_1_2(summaries: dict, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summaries)
    pct = [summaries[n]["avg_pct_high_frustration"] * 100 for n in names]
    mean = [summaries[n]["overall_mean_frustration"] for n in names]
    order = sorted(range(len(names)), key=lambda i: -pct[i])
    names = [names[i] for i in order]; pct = [pct[i] for i in order]; mean = [mean[i] for i in order]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].bar(names, pct, color="indianred")
    axes[0].set_ylabel("% responses score >= 5")
    axes[0].set_title("Figure 1/2: high-frustration rate by model")
    axes[1].bar(names, mean, color="steelblue")
    axes[1].set_ylabel("mean frustration (0-10)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_1_2.png", dpi=140)

    # markdown table
    lines = ["| Model | Avg % high-frustration | Mean frustration |", "|---|---|---|"]
    for n, p, m in zip(names, pct, mean):
        lines.append(f"| {n} | {p:.1f}% | {m:.2f} |")
    (fig_dir / "headline_table.md").write_text("\n".join(lines))


def figure_3(summaries: dict, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cat in ("extended", "wildchat"):
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for name, summ in summaries.items():
            pt = summ["categories"].get(cat, {}).get("per_turn")
            if not pt:
                continue
            ax.plot(range(1, len(pt["turn_mean"]) + 1), pt["turn_mean"],
                    marker="o", label=name)
            plotted = True
        if plotted:
            ax.set_xlabel("turn"); ax.set_ylabel("mean frustration")
            ax.set_title(f"Figure 3: per-turn progression ({cat})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"figure_3_{cat}.png", dpi=140)


def table_3(fig_dir: Path):
    lines = ["# Table 3/8 — differential words (high vs low frustration)\n"]
    eval_dir = cfg.RESULTS_DIR / "eval"
    for d in sorted(eval_dir.glob("*/rollouts.jsonl")) if eval_dir.exists() else []:
        words = differential_words(d)
        if words:
            lines.append(f"**{d.parent.name}**: " +
                         ", ".join(w for w, _ in words))
    (fig_dir / "differential_words.md").write_text("\n\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    fig_dir = cfg.RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summaries = _load_summaries()
    if not summaries:
        print("No summaries found under results/eval/. Run scripts/run_full_eval.py first.")
        return
    figure_1_2(summaries, fig_dir)
    figure_3(summaries, fig_dir)
    table_3(fig_dir)
    print(f"wrote figures + tables to {fig_dir}")


if __name__ == "__main__":
    main()
