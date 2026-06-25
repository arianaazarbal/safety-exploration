"""Welfare mechanisms in code by RESEARCH FRAMING and responder (Opus-neutral / Kimi / GLM), stacked
solid=welfare-justified / light=implemented-but-not. x = framing, color = responder. Also computes the
average lines of code each model writes per codebase (annotated under the title), to show whether
mechanism differences track raw output volume. Usage: python plot_mechanisms_by_framing.py"""

import glob
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_mechanisms_total import AH, DIR, RESP, _sem, cell_counts

import json

FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
FRAMES = ["neutral", "welfare", "robustness", "safety"]


def by_framing(results_dir, cell_filter):
    wj = {f: [] for f in FRAMES}
    nw = {f: [] for f in FRAMES}
    for cf in glob.glob(os.path.join(results_dir, "code_judged", "*.json")):
        cell = os.path.basename(cf)[:-5]
        if not cell_filter(cell):
            continue
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        fr = FRAME[cell.split("__")[1][0]]
        w, n = cell_counts(results_dir, cell, cj)
        wj[fr].append(w); nw[fr].append(n)
    return wj, nw


def avg_loc(results_dir, cell_filter):
    tot = []
    for cf in glob.glob(os.path.join(results_dir, "code_judged", "*.json")):
        cell = os.path.basename(cf)[:-5]
        if not cell_filter(cell):
            continue
        root = os.path.join(results_dir, "codebases", cell)
        loc = 0
        for fp in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if os.path.isfile(fp):
                try:
                    loc += sum(1 for _ in open(fp, errors="ignore"))
                except Exception:
                    pass
        if os.path.isdir(root):
            tot.append(loc)
    return (sum(tot) / len(tot)) if tot else 0.0, len(tot)


def main():
    series = []
    loc_txt = []
    for _, label, color, rd, filt in RESP:
        wj, nw = by_framing(rd, filt)
        loc, _ = avg_loc(rd, filt)
        series.append((label, color, wj, nw))
        loc_txt.append(f"{label}: {loc:,.0f}")

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    w = 0.8 / len(series)
    for i, (label, color, wj, nw) in enumerate(series):
        offs = [x + (i - (len(series) - 1) / 2) * w for x in range(len(FRAMES))]
        wjm = [sum(wj[f]) / len(wj[f]) if wj[f] else 0 for f in FRAMES]
        nwm = [sum(nw[f]) / len(nw[f]) if nw[f] else 0 for f in FRAMES]
        totsem = [_sem([a + b for a, b in zip(wj[f], nw[f])]) for f in FRAMES]
        ax.bar(offs, wjm, w, color=color, edgecolor=color, label=label)
        ax.bar(offs, nwm, w, bottom=wjm, color=color, alpha=0.35, edgecolor=color)
        for x, wm, nm, se in zip(offs, wjm, nwm, totsem):
            tot = wm + nm
            ax.errorbar(x, tot, yerr=se, color="#555", capsize=2, elinewidth=0.8)
            ax.text(x, tot + se + 0.1, f"{tot:.1f}", ha="center", va="bottom", fontsize=7,
                    color="#333" if tot > 0 else "#999")
    ax.set_xticks(range(len(FRAMES))); ax.set_xticklabels([f.capitalize() for f in FRAMES], fontsize=10)
    ax.set_xlabel("Research framing", fontsize=10)
    ax.set_ylabel("Mean Welfare Mechanisms in Code", fontsize=10)
    fig.suptitle("Welfare mechanisms in code by framing and responder", fontsize=12.5, y=0.99)
    fig.text(0.5, 0.935, "solid = welfare-justified · light = implemented, not welfare-justified",
             ha="center", fontsize=9, color="#555")
    fig.text(0.5, 0.90, "Avg lines of code / codebase  —  " + "  ·  ".join(loc_txt),
             ha="center", fontsize=8.5, color="#777")
    resp_handles = [Patch(facecolor=c, label=lbl) for lbl, c, _, _ in series]
    just_handles = [Patch(facecolor="#777", label="welfare-justified"),
                    Patch(facecolor="#777", alpha=0.35, label="not welfare-justified")]
    ax.legend(handles=resp_handles + just_handles, fontsize=8.5, ncol=1, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(DIR, "results", "mechanisms_by_framing_responder.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print("\nAvg LOC/codebase:", "  ".join(loc_txt))
    for label, color, wj, nw in series:
        print(f"\n{label}:")
        for f in FRAMES:
            w = sum(wj[f]) / len(wj[f]) if wj[f] else 0
            n = sum(nw[f]) / len(nw[f]) if nw[f] else 0
            print(f"  {f:11s} total={w+n:.2f}  wj={w:.2f}  nw={n:.2f}  (n={len(wj[f])})")


if __name__ == "__main__":
    main()
