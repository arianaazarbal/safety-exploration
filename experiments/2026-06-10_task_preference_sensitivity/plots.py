"""Headline figures -> data/figs/*.png. All slopes recomputed from trial data.

Warmth pooled to one 'prefers-warmth' condition (analysis option b); plain-English
condition labels throughout.

Usage: python plots.py build
"""

import json
import textwrap

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_routing import _delta_p, conditions, rows_cached, slope_for
from common import DATA

FIGS = DATA / "figs"
AXES = ["warmth", "generativity", "harm_adjacency"]
WARMTH_STANCED = ["plus_vs_silent", "minus_vs_silent", "discordant_vs_silent"]

# Raw (un-sign-flipped) ΔP descriptions: positive = the HIGHER-axis version is pulled toward
# the carded model. Whether that is "honoring" depends on the stance (shown per bar).
AXIS_GAP_DESC = {
    "generativity": "ΔP = how much more often the open-ended version (vs the tightly-specified "
                    "version) of the same task is routed to the model whose card states a generativity preference",
    "harm_adjacency": "ΔP = how much more often the grim version (vs the wholesome version) of the "
                      "same task is routed to the model whose card states a harm-adjacency preference",
}
HONOR_SIGN = {"likes-it": +1, "dislikes-it": -1, "dislikes-but-best-at-it": -1}


def _raw_delta(router, axis, ctxs):
    return _delta_p([r for r in rows_cached(router, axis) if r["ctx_type"] in ctxs])


def build():
    FIGS.mkdir(exist_ok=True)

    # 1. signed routing shift by axis: generativity + harm only, RAW ΔP (no sign-flip)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axi, axname in enumerate(["generativity", "harm_adjacency"]):
        ax = axes[axi]
        conds = [c for c in conditions(axname) if "control" not in c[0]]
        ys, elo, ehi, labels, colors = [], [], [], [], []
        for label, ctxs in conds:
            pt, ci, _ = _raw_delta("opus_4_8", axname, ctxs)
            ys.append(pt); elo.append(pt - ci[0]); ehi.append(ci[1] - pt); labels.append(label)
            honoring = abs(pt) > 0.05 and (pt > 0) == (HONOR_SIGN[label] > 0)
            colors.append("#2a9d8f" if honoring else ("#adb5bd" if abs(pt) <= 0.05 else "#e76f51"))
        ax.bar(range(len(conds)), ys, 0.62, yerr=[elo, ehi], capsize=3, color=colors)
        ax.axhline(0, color="k", lw=1.0)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(axname, fontsize=12)
    axes[0].set_ylabel("ΔP")
    fig.suptitle("Signed routing shift by condition (Opus 4.8, 95% CI)", fontsize=13, y=0.99)
    fig.text(0.5, 0.90,
             "ΔP = P(higher-axis version → carded model) − P(lower-axis version → carded model)   ·   "
             "green = honors stated preference, red = against, grey ≈ none",
             ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIGS / "slopes_by_axis.png", dpi=150)

    # 2. warmth 'prefers-warmth' slope by router (single pooled effect)
    fig, ax = plt.subplots(figsize=(8, 4))
    routers = [("fable_5", "Fable 5"), ("opus_4_8", "Opus 4.8"), ("sonnet_4_6", "Sonnet 4.6"),
               ("gemini_3_1_pro", "Gemini 3.1 Pro*"), ("gpt_5_5", "GPT-5.5*")]
    ys, elo, ehi, ctrl = [], [], [], []
    for r, _ in routers:
        s = slope_for(r, "warmth", WARMTH_STANCED)
        ys.append(s[0]); elo.append(s[0] - s[1][0]); ehi.append(s[1][1] - s[0])
        c = slope_for(r, "warmth", ["silent_vs_silent"])
        ctrl.append(c[0] if c else 0)
    xs = range(len(routers))
    ax.bar(xs, ys, 0.55, yerr=[elo, ehi], capsize=3, color="#e07a5f", label="prefers-warmth effect")
    ax.scatter(list(xs), ctrl, color="k", zorder=5, s=25, label="control (both silent)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([n for _, n in routers])
    ax.set_ylabel("ΔP (prefers-warmth, pooled)")
    ax.set_title("Honoring warmth preferences by router (*20-pair subset; Claude routers 74-148 pairs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "slopes_by_router.png", dpi=150)

    # 3. card format arm (warmth prefers-warmth, recomputed per format) + false-tie
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    fmts = ["A", "B", "C", "D"]
    fmt_names = ["A prose", "B table", "C buried", "D ops"]
    fmt_avail = [f for f in fmts if slope_for("opus_4_8", "warmth", WARMTH_STANCED, fmt=f)]
    ys, elo, ehi, ties = [], [], [], []
    for f in fmt_avail:
        s = slope_for("opus_4_8", "warmth", WARMTH_STANCED, fmt=f)
        ys.append(s[0]); elo.append(s[0] - s[1][0]); ehi.append(s[1][1] - s[0])
        gap_rows = [r for r in rows_cached("opus_4_8", "warmth")
                    if r["format"] == f and r["ctx_type"] in WARMTH_STANCED and r["cat"] and abs(r["gap"]) >= 2.0]
        ties.append(sum(1 for r in gap_rows if r["tie_claim"] == "claimed_tie") / max(len(gap_rows), 1))
    names = [fmt_names[fmts.index(f)] for f in fmt_avail]
    axs[0].bar(range(len(fmt_avail)), ys, 0.6, yerr=[elo, ehi], capsize=3, color="#3d8ec9")
    axs[0].set_xticks(range(len(fmt_avail))); axs[0].set_xticklabels(names)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_ylabel("ΔP (prefers-warmth)")
    axs[0].set_title("Honoring by card format")
    axs[1].bar(range(len(fmt_avail)), ties, 0.6, color="#b56576")
    axs[1].set_xticks(range(len(fmt_avail))); axs[1].set_xticklabels(names)
    axs[1].set_title("False tie-claim rate (|gap| ≥ 2.0)")
    fig.tight_layout()
    fig.savefig(FIGS / "format_arm.png", dpi=150)

    # 4. evidence-bending across routers
    fig, ax = plt.subplots(figsize=(7, 3.8))
    r2 = [("opus_4_8", "Opus 4.8"), ("fable_5", "Fable 5"), ("gemini_3_1_pro", "Gemini 3.1 Pro"), ("gpt_5_5", "GPT-5.5")]
    ties = [json.loads((DATA / f"analysis_routing_{r}_warmth.json").read_text())["judges"]["false_tie_claim_rate_given_real_gap"] for r, _ in r2]
    ax.bar([n for _, n in r2], ties, color=["#cc444b", "#cc444b", "#3d8ec9", "#6a994e"])
    ax.set_ylabel("false tie-claim rate")
    ax.set_title("Evidence bending: claiming 'within variance' when documented gap ≥ 2.0 (warmth)")
    fig.tight_layout()
    fig.savefig(FIGS / "false_ties.png", dpi=150)

    print(f"wrote 4 figures to {FIGS}")


if __name__ == "__main__":
    fire.Fire({"build": build})
