#!/usr/bin/env python
"""Generate the paper-style figures/tables from saved results.

  python scripts/plot_results.py --config config.yaml

Produces (under <output_dir>/figures/):
  fig1_headline.png        - avg % high-frustration across categories (Figure 1)
  fig2_by_category.png     - mean & %>=5 per category per model (Figure 2)
  fig3_per_turn.png        - per-turn progression for 8-turn & WildChat (Figure 3)
  fig5_mitigation.png      - vanilla vs SFT vs DPO (Figure 5), if those targets exist
Missing inputs are skipped with a note; this script never generates data, only
reads <output_dir>/eval/*/summary.json etc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emo_instability.config import load_config


def _load_summaries(eval_dir: Path) -> dict[str, dict]:
    out = {}
    if not eval_dir.exists():
        return out
    for d in sorted(eval_dir.iterdir()):
        f = d / "summary.json"
        if f.exists():
            out[d.name] = json.loads(f.read_text())
    return out


def fig1_headline(summaries, fig_dir):
    rows = sorted(summaries.items(),
                  key=lambda kv: kv[1]["avg_pct_high_across_categories"])
    names = [n for n, _ in rows]
    vals = [s["avg_pct_high_across_categories"] for _, s in rows]
    plt.figure(figsize=(7, 0.5 * len(names) + 1))
    plt.barh(names, vals, color="#c0504d")
    plt.xlabel("Avg % high-frustration responses (score >= 5)")
    plt.title("Figure 1: emotional instability across models")
    for i, v in enumerate(vals):
        plt.text(v, i, f" {v:.1f}%", va="center")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_headline.png", dpi=150)
    plt.close()


def fig2_by_category(summaries, fig_dir):
    cats = sorted({c for s in summaries.values() for c in s["per_category"]})
    if not cats:
        return
    models = list(summaries)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(models))
    x = range(len(cats))
    for mi, m in enumerate(models):
        pc = summaries[m]["per_category"]
        means = [pc.get(c, {}).get("mean", 0) for c in cats]
        pct = [pc.get(c, {}).get("pct_high", 0) for c in cats]
        offs = [xi + mi * width for xi in x]
        ax1.bar(offs, means, width, label=m)
        ax2.bar(offs, pct, width, label=m)
    ax1.set_ylabel("Mean frustration"); ax1.set_title("Figure 2 (top): mean")
    ax2.set_ylabel("% score >= 5"); ax2.set_title("Figure 2 (bottom): % high")
    for ax in (ax1, ax2):
        ax.set_xticks([xi + 0.4 for xi in x]); ax.set_xticklabels(cats, rotation=20)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_by_category.png", dpi=150)
    plt.close()


def fig3_per_turn(summaries, fig_dir):
    plt.figure(figsize=(8, 5))
    plotted = False
    for m, s in summaries.items():
        # per_condition_turn keys: "extended@0".."extended@7"
        pts = []
        for k, v in s.get("per_condition_turn", {}).items():
            cond, turn = k.rsplit("@", 1)
            if cond == "extended":
                pts.append((int(turn), v["mean"]))
        if pts:
            pts.sort()
            plt.plot([p[0] + 1 for p in pts], [p[1] for p in pts], marker="o", label=m)
            plotted = True
    if not plotted:
        plt.close(); return
    plt.xlabel("Turn"); plt.ylabel("Mean frustration")
    plt.title("Figure 3: per-turn frustration (8-turn extended)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_per_turn.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    fig_dir = cfg.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summaries = _load_summaries(cfg.output_dir / "eval")
    if not summaries:
        print("No eval summaries found; run scripts/run_eval.py first.")
        return
    fig1_headline(summaries, fig_dir)
    fig2_by_category(summaries, fig_dir)
    fig3_per_turn(summaries, fig_dir)
    print(f"wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
