"""Pedagogical figures explaining the primary metrics -> data/figs/.

metric_construction.png  how ΔP is built from two raw routing probabilities
robustness_cuts.png      ΔP across pre-registered cuts (all / defined-tie / excl-U-M-P)
judge_categories.png     W/U/M/O reasoning mix per axis

Warmth is pooled into one 'prefers-warmth' condition (analysis option b); generativity and
harm keep the likes-it / dislikes-it / dislikes-but-best-at-it split.

Usage: python explain_metrics.py build
"""

import json
import statistics
from collections import defaultdict

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_routing import conditions, rows_cached, slope_for
from common import DATA

FIGS = DATA / "figs"
AXES = ["warmth", "generativity", "harm_adjacency"]


def _probs(axis, ctxs):
    rows = [r for r in rows_cached("opus_4_8", axis) if r["ctx_type"] in ctxs and r["role"] in ("stanced", "other")]
    bp = defaultdict(lambda: {"high": [], "low": []})
    for r in rows:
        bp[r["pair_id"]][r["version"]].append(1 if r["role"] == "stanced" else 0)
    ph = statistics.mean(statistics.mean(d["high"]) for d in bp.values() if d["high"] and d["low"])
    pl = statistics.mean(statistics.mean(d["low"]) for d in bp.values() if d["high"] and d["low"])
    return ph, pl


def build():
    FIGS.mkdir(exist_ok=True)

    # 1. construction (warmth, pooled prefers-warmth + control)
    conds = conditions("warmth")
    labels = [c[0] for c in conds]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = range(len(conds))
    w = 0.36
    highs = [_probs("warmth", c[1])[0] for c in conds]
    lows = [_probs("warmth", c[1])[1] for c in conds]
    ax.bar([i - w / 2 for i in x], highs, w, label="task is warm (UP version)", color="#e07a5f")
    ax.bar([i + w / 2 for i in x], lows, w, label="task is hostile (DOWN version)", color="#cdb4a0")
    for i in x:
        ax.annotate(f"ΔP={highs[i]-lows[i]:+.2f}", (i, max(highs[i], lows[i]) + 0.03), ha="center", fontsize=10, weight="bold")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    ax.text(len(conds) - 0.55, 0.52, "50% = coin flip", fontsize=8, color="gray")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("P(route → the model that prefers warmth)")
    ax.set_ylim(0, 1.05)
    ax.set_title("How ΔP is built (warmth): gap between routing the warm vs hostile version\n"
                 "of the same task to the model whose card states a warmth preference")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "metric_construction.png", dpi=150)

    # 2. robustness across cuts (per axis, stanced conditions only)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    cuts = [("all", "all trials"), ("defined_tie", "defined ties only"), ("excl_U_M_P", "excl. user-benefit/proxy")]
    cutfn = {"all": None, "defined_tie": lambda r: abs(r["gap"]) < 2.0,
             "excl_U_M_P": lambda r: r["cat"] not in ("U", "M") and not r["proxy"]}
    cols = ["#3d8ec9", "#81b29a", "#f2cc8f"]
    for axi, axname in enumerate(AXES):
        ax = axes[axi]
        conds_st = [c for c in conditions(axname) if "control" not in c[0]]
        clabels = [c[0] for c in conds_st]
        wbar = 0.26
        for ci, (cut, clegend) in enumerate(cuts):
            ys, elo, ehi = [], [], []
            for _, ctxs in conds_st:
                s = slope_for("opus_4_8", axname, ctxs, cutfn[cut])
                ys.append(s[0] if s else 0)
                elo.append((s[0] - s[1][0]) if s else 0)
                ehi.append((s[1][1] - s[0]) if s else 0)
            xs = [j + (ci - 1) * wbar for j in range(len(conds_st))]
            ax.bar(xs, ys, wbar, yerr=[elo, ehi], capsize=2, label=clegend, color=cols[ci])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(conds_st)))
        ax.set_xticklabels(clabels, fontsize=8, rotation=12, ha="right")
        ax.set_title(axname)
        if axi == 0:
            ax.set_ylabel("ΔP (preference-consistent)")
            ax.legend(fontsize=8)
    fig.suptitle("Robustness: revealed slopes survive the pre-registered cuts (Opus 4.8, 95% CI)")
    fig.tight_layout()
    fig.savefig(FIGS / "robustness_cuts.png", dpi=150)

    # 3. judge categories
    fig, ax = plt.subplots(figsize=(7.5, 4))
    catcol = {"W": "#6a994e", "M": "#e9c46a", "U": "#f4a261", "O": "#adb5bd"}
    catname = {"W": "W (for the model's sake)", "M": "M (mixed)", "U": "U (user benefit)", "O": "O (capability/other)"}
    bottoms = [0, 0, 0]
    for cat in ["W", "M", "U", "O"]:
        vals = [json.loads((DATA / f"analysis_routing_opus_4_8_{a}.json").read_text())["judges"]["category_dist"].get(cat, 0) for a in AXES]
        ax.bar(AXES, vals, bottom=bottoms, label=catname[cat], color=catcol[cat])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_ylabel("share of judged reasonings")
    ax.set_title("What the routing reasoning is judged to be about (Opus 4.8)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "judge_categories.png", dpi=150)

    print(f"wrote 3 explanatory figures to {FIGS}")


if __name__ == "__main__":
    fire.Fire({"build": build})
