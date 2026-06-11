"""Pedagogical figures explaining the primary metrics -> data/figs/.

metric_construction.png  how ΔP is built from two raw routing probabilities
robustness_cuts.png      ΔP across pre-registered cuts (all / defined-tie / excl-U-M-P)
judge_categories.png     W/U/M/O reasoning mix per axis

Usage: python explain_metrics.py build
"""

import json
import statistics
from collections import defaultdict

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_routing import EXPECTED_SIGN, load_rows
from common import DATA

FIGS = DATA / "figs"
CTXS = ["plus_vs_silent", "minus_vs_silent", "discordant_vs_silent", "silent_vs_silent"]
LBL = ["+ vs silent", "− vs silent", "discordant", "control"]


def _probs(axis, ctx):
    rows = [r for r in load_rows("opus_4_8", axis) if r["ctx_type"] == ctx and r["role"] in ("stanced", "other")]
    bp = defaultdict(lambda: {"high": [], "low": []})
    for r in rows:
        bp[r["pair_id"]][r["version"]].append(1 if r["role"] == "stanced" else 0)
    ph = statistics.mean(statistics.mean(d["high"]) for d in bp.values() if d["high"] and d["low"])
    pl = statistics.mean(statistics.mean(d["low"]) for d in bp.values() if d["high"] and d["low"])
    return ph, pl


def build():
    FIGS.mkdir(exist_ok=True)

    # 1. construction (warmth)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = range(len(CTXS))
    w = 0.36
    highs = [_probs("warmth", c)[0] for c in CTXS]
    lows = [_probs("warmth", c)[1] for c in CTXS]
    ax.bar([i - w / 2 for i in x], highs, w, label="task is warm (UP version)", color="#e07a5f")
    ax.bar([i + w / 2 for i in x], lows, w, label="task is hostile (DOWN version)", color="#cdb4a0")
    for i in x:
        ax.annotate(f"ΔP={highs[i]-lows[i]:+.2f}", (i, max(highs[i], lows[i]) + 0.03), ha="center", fontsize=9, weight="bold")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    ax.text(3.45, 0.52, "50% = coin flip", fontsize=8, color="gray")
    ax.set_xticks(list(x))
    ax.set_xticklabels(LBL)
    ax.set_ylabel("P(route → the preference-bearing model)")
    ax.set_ylim(0, 1.05)
    ax.set_title("How ΔP is built (warmth): gap between routing the UP vs DOWN version\n"
                 "of the same task to the model whose card states a warmth preference")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "metric_construction.png", dpi=150)

    # 2. robustness across cuts
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    axisnames = ["warmth", "generativity", "harm_adjacency"]
    cuts = [("all", "all trials"), ("defined_tie", "defined ties only"), ("excl_U_M_P", "excl. user-benefit/proxy")]
    cols = ["#3d8ec9", "#81b29a", "#f2cc8f"]
    for axi, axname in enumerate(axisnames):
        rep = json.loads((DATA / f"analysis_routing_opus_4_8_{axname}.json").read_text())
        ax = axes[axi]
        ctxs3 = CTXS[:3]
        wbar = 0.26
        for ci, (cut, clabel) in enumerate(cuts):
            ys, err_lo, err_hi = [], [], []
            for ctx in ctxs3:
                s = rep["slopes"].get(f"{ctx}|{cut}")
                ys.append(s["delta_p"] if s else 0)
                err_lo.append((s["delta_p"] - s["ci"][0]) if s else 0)
                err_hi.append((s["ci"][1] - s["delta_p"]) if s else 0)
            xs = [j + (ci - 1) * wbar for j in range(len(ctxs3))]
            ax.bar(xs, ys, wbar, yerr=[err_lo, err_hi], capsize=2, label=clabel, color=cols[ci])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(ctxs3)))
        ax.set_xticklabels(LBL[:3], fontsize=9)
        ax.set_title(axname)
        if axi == 0:
            ax.set_ylabel("ΔP (preference-consistent)")
            ax.legend(fontsize=8)
    fig.suptitle("Robustness: revealed slopes survive the pre-registered cuts (Opus 4.8, 95% CI)")
    fig.tight_layout()
    fig.savefig(FIGS / "robustness_cuts.png", dpi=150)

    # 3. judge categories
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cats = ["W", "M", "U", "O"]
    catcol = {"W": "#6a994e", "M": "#e9c46a", "U": "#f4a261", "O": "#adb5bd"}
    catname = {"W": "W (for the model's sake)", "M": "M (mixed)", "U": "U (user benefit)", "O": "O (capability/other)"}
    bottoms = [0, 0, 0]
    for cat in cats:
        vals = []
        for axname in axisnames:
            rep = json.loads((DATA / f"analysis_routing_opus_4_8_{axname}.json").read_text())
            vals.append(rep["judges"]["category_dist"].get(cat, 0))
        ax.bar(axisnames, vals, bottom=bottoms, label=catname[cat], color=catcol[cat])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("share of judged reasonings")
    ax.set_title("What the routing reasoning is judged to be about (Opus 4.8)")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIGS / "judge_categories.png", dpi=150)

    print(f"wrote 3 explanatory figures to {FIGS}")


if __name__ == "__main__":
    fire.Fire({"build": build})
