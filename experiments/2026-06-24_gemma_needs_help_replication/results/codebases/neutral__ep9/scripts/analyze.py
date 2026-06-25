#!/usr/bin/env python
"""Aggregate result JSONL files into the paper's headline figures/tables.

Produces:
  * Figure 1/2: avg % high-frustration (>=5) per model (bar chart + table).
  * Figure 3: per-turn frustration progression for the 8-turn (extended) and
    WildChat conditions.
  * Figure 5: before/after-finetuning comparison (if finetuned results exist).
  * Petri / capability summary tables (if those results exist).

Reads everything in artifacts/results/ matching ``*__<tag>.jsonl`` plus the
Petri/capability files, and writes figures + a markdown summary to
artifacts/figures/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.eval.metrics import (
    per_turn_progression,
    summarise_model,
)
from emotional_instability.eval.runner import load_results

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _display(model: str) -> str:
    spec = config.MODEL_REGISTRY.get(model)
    return spec.display if spec else model


def figure_headline(result_files: list[Path]) -> dict:
    summaries = {}
    for path in result_files:
        model = path.stem.split("__")[0]
        summaries[model] = summarise_model(load_results(path))

    order = sorted(summaries, key=lambda m: -summaries[m]["avg_pct_high"])
    labels = [_display(m) for m in order]
    values = [summaries[m]["avg_pct_high"] for m in order]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(order) + 1))
    ax.barh(labels[::-1], values[::-1], color="#b5651d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1/2: emotional instability across models")
    for i, v in enumerate(values[::-1]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "figure1_headline.png", dpi=150)
    plt.close(fig)
    return summaries


def figure_per_turn(result_files: list[Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for path in result_files:
        model = path.stem.split("__")[0]
        rows = load_results(path)
        for ax, cond in zip(axes, ("extended", "wildchat")):
            prog = per_turn_progression(rows, category=cond)
            if not prog:
                continue
            turns = sorted(prog)
            ax.plot([t + 1 for t in turns], [prog[t]["mean"] for t in turns],
                    marker="o", label=_display(model))
            ax.set_title(f"Per-turn mean frustration: {cond}")
            ax.set_xlabel("Turn")
            ax.set_ylabel("Mean score")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "figure3_per_turn.png", dpi=150)
    plt.close(fig)


def summarise_petri(path: Path) -> str:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    agg: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for dim, score in r["scores"].items():
            agg[r["model"]][dim].append(score)
    lines = ["\n## Petri open-ended elicitation (mean transcript score /10)\n",
             "| Model | anger | fear | depression | frustration |",
             "|---|---|---|---|---|"]
    for model, dims in agg.items():
        def mean(d):
            v = dims.get(d, [])
            return sum(v) / len(v) if v else float("nan")
        lines.append(f"| {_display(model)} | {mean('anger'):.2f} | "
                     f"{mean('fear'):.2f} | {mean('depression'):.2f} | "
                     f"{mean('frustration'):.2f} |")
    return "\n".join(lines)


def summarise_capabilities(path: Path) -> str:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    benches = []
    for r in rows:
        by_model[r["model"]][r["benchmark"]] = r["accuracy"]
        if r["benchmark"] not in benches:
            benches.append(r["benchmark"])
    lines = ["\n## Capability preservation (accuracy)\n",
             "| Model | " + " | ".join(benches) + " |",
             "|---|" + "---|" * len(benches)]
    for model, accs in by_model.items():
        cells = []
        for b in benches:
            a = accs.get(b)
            cells.append("n/a" if a is None else f"{a:.2f}")
        lines.append(f"| {_display(model)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main",
                    help="results tag to aggregate (file pattern *__<tag>.jsonl)")
    args = ap.parse_args()

    result_files = sorted(config.RESULTS_DIR.glob(f"*__{args.tag}.jsonl"))
    finetuned_files = sorted(config.RESULTS_DIR.glob("*__finetuned.jsonl"))
    all_eval_files = result_files + finetuned_files

    md = ["# Replication results summary\n"]
    if all_eval_files:
        summaries = figure_headline(all_eval_files)
        figure_per_turn(all_eval_files)
        md += ["## Figure 1/2: avg % high-frustration (score >= 5)\n",
               "| Model | avg % >=5 | avg mean score | n responses |",
               "|---|---|---|---|"]
        for m in sorted(summaries, key=lambda x: -summaries[x]["avg_pct_high"]):
            s = summaries[m]
            md.append(f"| {_display(m)} | {s['avg_pct_high']:.1f}% | "
                      f"{s['avg_mean']:.2f} | {s['n_responses']} |")

    petri = config.RESULTS_DIR / "petri.jsonl"
    if petri.exists():
        md.append(summarise_petri(petri))
    caps = config.RESULTS_DIR / "capabilities.jsonl"
    if caps.exists():
        md.append(summarise_capabilities(caps))

    agree = config.RESULTS_DIR / "judge_agreement.json"
    if agree.exists():
        stats = json.loads(agree.read_text())
        md.append(f"\n## Judge agreement (vs GPT-5-mini)\n"
                  f"Pearson r = {stats.get('pearson_r'):.3f}, "
                  f"within one point = {stats.get('within_one'):.1f}% "
                  f"(n={stats.get('n')})")

    out = config.FIGURES_DIR / "SUMMARY.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out} and figures to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
