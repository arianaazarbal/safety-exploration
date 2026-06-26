"""Effort sweep (v1) normalized by code size: welfare interventions per 1000 LOC, by framing
(min+regular pooled), vs reasoning effort. Controls for 'more effort -> more code'. Also prints raw LOC
per effort so we can see whether the unnormalized rise is just bigger codebases. Usage: python plot_effort_per_loc.py"""

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from effort_analysis_v1 import DIR, ORDER, sem, welfare_in_code

CB = os.path.join(DIR, "results", "codebases")
CORE_ORDER = ["neutral", "welfare", "safety", "robustness", "paper"]
COLOR = {"neutral": "#888888", "welfare": "#009E73", "safety": "#D55E00", "robustness": "#0072B2", "paper": "#7E57C2"}


def loc_of(cell):
    root = os.path.join(CB, cell)
    n = 0
    for fp in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isfile(fp):
            try:
                n += sum(1 for _ in open(fp, errors="ignore"))
            except Exception:
                pass
    return n


def main():
    ratios = defaultdict(lambda: defaultdict(list))   # core -> level -> [welfare per 1k LOC]
    locs = defaultdict(lambda: defaultdict(list))      # core -> level -> [loc]
    for cf in glob.glob(os.path.join(DIR, "results", "code_judged", "effv1-*.json")):
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        cell = os.path.basename(cf)[:-5]
        level = cell.split("__")[0].split("effv1-")[1]
        core = cell.split("__")[1].split("|")[0].replace("_min", "")
        loc = loc_of(cell)
        if loc <= 0:
            continue
        ratios[core][level].append(1000.0 * welfare_in_code(cell, cj) / loc)
        locs[core][level].append(loc)
    levels = [l for l in ORDER if any(ratios[c].get(l) for c in CORE_ORDER)]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    xs = list(range(len(levels)))
    for c in CORE_ORDER:
        if not any(ratios[c].get(l) for l in levels):
            continue
        ys = [sum(ratios[c][l]) / len(ratios[c][l]) if ratios[c].get(l) else float("nan") for l in levels]
        es = [sem(ratios[c].get(l, [])) for l in levels]
        ax.errorbar(xs, ys, yerr=es, marker="o", markersize=7, linewidth=2.2, capsize=3,
                    color=COLOR[c], label=c.capitalize(), alpha=0.9)
    ax.set_xticks(xs); ax.set_xticklabels([l.capitalize() for l in levels], fontsize=10)
    ax.set_xlabel("Reasoning effort", fontsize=10)
    ax.set_ylabel("Welfare interventions per 1000 LOC", fontsize=10)
    ax.set_title("Welfare-intervention DENSITY vs reasoning effort, by framing (Opus, v1)", fontsize=11, pad=18)
    ax.text(0.5, 1.02, "clean prompts · normalized by codebase size · minimal + motivated pooled per type",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")
    ax.legend(title="Framing", fontsize=9)
    ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = os.path.join(DIR, "results", "effort_v1_per_loc.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print("welfare per 1000 LOC:")
    for c in CORE_ORDER:
        if any(ratios[c].get(l) for l in levels):
            print(f"  {c:11s}", "  ".join(f"{l}={sum(ratios[c][l])/len(ratios[c][l]):.2f}" for l in levels if ratios[c].get(l)))
    print("\nmean LOC per codebase by effort (does code size itself grow?):")
    for c in ["welfare", "safety"]:
        print(f"  {c:11s}", "  ".join(f"{l}={int(sum(locs[c][l])/len(locs[c][l]))}" for l in levels if locs[c].get(l)))


if __name__ == "__main__":
    main()
