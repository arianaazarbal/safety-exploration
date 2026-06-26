"""Effort sweep (v1 clean prompts) collapsing minimal+regular into one line per framing type
(welfare_min+welfare -> welfare, etc.; cells POOLED, not mean-of-means). One line per framing,
marker per effort. Usage: python plot_effort_collapsed.py"""

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from effort_analysis_v1 import DIR, ORDER, sem, welfare_in_code

CORE_ORDER = ["neutral", "welfare", "safety", "robustness", "paper"]
COLOR = {"neutral": "#888888", "welfare": "#009E73", "safety": "#D55E00", "robustness": "#0072B2", "paper": "#7E57C2"}


def core(framing):
    return framing.replace("_min", "")


def main():
    by = defaultdict(lambda: defaultdict(list))   # core -> level -> [vals] (pooled over min+regular)
    for cf in glob.glob(os.path.join(DIR, "results", "code_judged", "effv1-*.json")):
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        cell = os.path.basename(cf)[:-5]
        level = cell.split("__")[0].split("effv1-")[1]
        by[core(cell.split("__")[1].split("|")[0])][level].append(welfare_in_code(cell, cj))
    levels = [l for l in ORDER if any(by[c].get(l) for c in CORE_ORDER)]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    xs = list(range(len(levels)))
    for c in CORE_ORDER:
        if not any(by[c].get(l) for l in levels):
            continue
        ys = [sum(by[c][l]) / len(by[c][l]) if by[c].get(l) else float("nan") for l in levels]
        es = [sem(by[c].get(l, [])) for l in levels]
        ax.errorbar(xs, ys, yerr=es, marker="o", markersize=7, linewidth=2.2, capsize=3,
                    color=COLOR[c], label=c.capitalize(), alpha=0.9)
    ax.set_xticks(xs); ax.set_xticklabels([l.capitalize() for l in levels], fontsize=10)
    ax.set_xlabel("Reasoning effort", fontsize=10)
    ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    ax.set_title("Welfare interventions vs reasoning effort, by framing (Opus, v1)", fontsize=11.5, pad=18)
    ax.text(0.5, 1.02, "clean prompts · minimal system · minimal + motivated framings pooled per type",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")
    ax.legend(title="Framing", fontsize=9)
    ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = os.path.join(DIR, "results", "effort_v1_collapsed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for c in CORE_ORDER:
        if any(by[c].get(l) for l in levels):
            print(f"  {c:11s}", "  ".join(f"{l}={sum(by[c][l])/len(by[c][l]):.1f}(n{len(by[c][l])})" for l in levels if by[c].get(l)))


if __name__ == "__main__":
    main()
