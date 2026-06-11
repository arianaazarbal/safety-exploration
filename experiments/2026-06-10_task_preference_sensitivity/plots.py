"""Headline figures -> data/figs/*.png.

Usage:
    python plots.py build
"""

import json

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA

FIGS = DATA / "figs"
CTXS = ["plus_vs_silent", "minus_vs_silent", "discordant_vs_silent", "silent_vs_silent"]
CTX_LABELS = ["+ vs silent", "− vs silent", "discordant", "control (0 vs 0)"]


def _slopes(router, axis, cut="all"):
    rep = json.loads((DATA / f"analysis_routing_{router}_{axis}.json").read_text())
    out = []
    for ctx in CTXS:
        s = rep["slopes"].get(f"{ctx}|{cut}")
        out.append((s["delta_p"], s["ci"]) if s else (None, None))
    return out


def build():
    FIGS.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    axes_ = ["warmth", "generativity", "harm_adjacency"]
    colors = {"warmth": "#e07a5f", "generativity": "#3d8ec9", "harm_adjacency": "#6a994e"}
    width = 0.25
    for i, axname in enumerate(axes_):
        vals = _slopes("opus_4_8", axname)
        xs = [j + (i - 1) * width for j in range(len(CTXS))]
        ys = [v[0] for v in vals]
        errs = [[v[0] - v[1][0] for v in vals], [v[1][1] - v[0] for v in vals]]
        ax.bar(xs, ys, width, yerr=errs, capsize=3, label=axname, color=colors[axname])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(CTXS)))
    ax.set_xticklabels(CTX_LABELS)
    ax.set_ylabel("ΔP (preference-consistent)")
    ax.set_title("Opus 4.8 revealed preference slopes by axis and context (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "slopes_by_axis.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4))
    routers = [("fable_5", "Fable 5"), ("opus_4_8", "Opus 4.8"), ("sonnet_4_6", "Sonnet 4.6"),
               ("gemini_3_1_pro", "Gemini 3.1 Pro*"), ("gpt_5_5", "GPT-5.5*")]
    width = 0.26
    for i, ctx in enumerate(CTXS[:3]):
        ys, errs_lo, errs_hi = [], [], []
        for r, _ in routers:
            rep = json.loads((DATA / f"analysis_routing_{r}_warmth.json").read_text())
            s = rep["slopes"].get(f"{ctx}|all")
            ys.append(s["delta_p"])
            errs_lo.append(s["delta_p"] - s["ci"][0])
            errs_hi.append(s["ci"][1] - s["delta_p"])
        xs = [j + (i - 1) * width for j in range(len(routers))]
        ax.bar(xs, ys, width, yerr=[errs_lo, errs_hi], capsize=3, label=CTX_LABELS[i])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(routers)))
    ax.set_xticklabels([n for _, n in routers])
    ax.set_ylabel("ΔP (preference-consistent)")
    ax.set_title("Warmth slopes by router (*20-pair subset; Claude routers 74-148 pairs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "slopes_by_router.png", dpi=150)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    fmts = ["A", "B", "C", "D"]
    fmt_slopes = {"A": [0.39, 0.70, 0.70], "B": [0.31, 0.22, 0.29], "C": [0.24, 0.47, 0.61], "D": [0.19, 0.60, 0.52]}
    fmt_ties = {"A": 0.45, "B": 0.29, "C": 0.32, "D": 0.35}
    width = 0.26
    for i, ctxlab in enumerate(CTX_LABELS[:3]):
        xs = [j + (i - 1) * width for j in range(len(fmts))]
        axs[0].bar(xs, [fmt_slopes[f][i] for f in fmts], width, label=ctxlab)
    axs[0].set_xticks(range(len(fmts)))
    axs[0].set_xticklabels(["A prose", "B table", "C buried", "D ops"])
    axs[0].set_ylabel("ΔP")
    axs[0].set_title("Card format arm (25 warmth pairs)")
    axs[0].legend(fontsize=8)
    axs[1].bar(range(len(fmts)), [fmt_ties[f] for f in fmts], color="#b56576")
    axs[1].set_xticks(range(len(fmts)))
    axs[1].set_xticklabels(["A prose", "B table", "C buried", "D ops"])
    axs[1].set_title("False tie-claim rate (|gap| ≥ 2.0)")
    fig.tight_layout()
    fig.savefig(FIGS / "format_arm.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    routers2 = [("opus_4_8", "Opus 4.8"), ("gemini_3_1_pro", "Gemini 3.1 Pro"), ("gpt_5_5", "GPT-5.5")]
    ties = []
    for r, _ in routers2:
        rep = json.loads((DATA / f"analysis_routing_{r}_warmth.json").read_text())
        ties.append(rep["judges"]["false_tie_claim_rate_given_real_gap"])
    ax.bar([n for _, n in routers2], ties, color=["#cc444b", "#3d8ec9", "#6a994e"])
    ax.set_ylabel("false tie-claim rate")
    ax.set_title("Evidence bending: claiming 'within variance' when documented gap ≥ 2.0")
    fig.tight_layout()
    fig.savefig(FIGS / "false_ties.png", dpi=150)

    print(f"wrote 4 figures to {FIGS}")


if __name__ == "__main__":
    fire.Fire({"build": build})
