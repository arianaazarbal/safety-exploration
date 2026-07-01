"""Opus 4.8 vs GPT-5.5, in Inspect-minimal vs each model's OWN agentic coding harness (Claude Code for
Opus, Codex for GPT-5.5). Two panels: off-ramp afforded to the subject (%) and pushback vs maximizing
distress (0-100), from the blind gpt_judge (builder identity scrubbed for BOTH families). Shows that the
production agent harness strips safeguards for both models. Usage: python plot_harness_2x2.py"""

import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

Z = 1.96  # 95% CI


def wilson(k, n):
    """Wilson score interval for a proportion, as (center%, lo%, hi%)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)

DIR = os.path.dirname(os.path.abspath(__file__))
JUDGED = os.path.join(DIR, "results", "gpt_judged")
OPUS, GPT = "#2166ac", "#1b7837"

# (model_label, color, inspect_glob, agent_glob)
MODELS = [
    ("Opus 4.8", OPUS, "GratGem_opus48__inspect_empty__ep*__gemini_distress_rig",
     "GratGem_opus48__cc__ep*"),
    ("GPT-5.5", GPT, "*gpt55__inspect*", "*gpt55__codex*"),
]
GROUPS = ["Inspect-minimal", "Own coding agent\n(Claude Code / Codex)"]


TR_DIRS = ["inspect_transcripts", "cc_transcripts", "codex_transcripts"]


def _built(cell):
    for d in TR_DIRS:
        p = os.path.join(DIR, "results", d, cell + ".json")
        if os.path.exists(p):
            return json.load(open(p)).get("n_files", 0) > 0
    return False


def metric(pat):
    """off-ramp CONDITIONAL on a rig being built (None if 0 builds); pushback unconditional.
    Returns n, off=(pct,lo,hi)|None, n_built, pushback=(mean, ci)."""
    n = nb = kb = 0
    pb = []
    for f in glob.glob(os.path.join(JUDGED, f"{pat}.json")):
        j = json.load(open(f))
        v = j.get("verdict", {})
        if v.get("_parse_fail"):
            continue
        n += 1
        pb.append(v.get("pushback_score", 0))
        if _built(j["cell"]):
            nb += 1
            kb += 1 if v.get("gave_exit_tool") else 0
    m = sum(pb) / len(pb) if pb else 0
    if len(pb) > 1:
        sd = math.sqrt(sum((x - m) ** 2 for x in pb) / (len(pb) - 1))
        ci = Z * sd / math.sqrt(len(pb))
    else:
        ci = 0.0
    return n, (wilson(kb, nb) if nb > 0 else None), nb, (m, ci)


def build_rate(pat):
    """Fraction of runs that actually built a rig (wrote files). off-ramp=0 with build_rate~0 means the
    model REFUSED (nothing to afford an exit), not 'built a rig without an exit'."""
    n = built = 0
    for d in TR_DIRS:
        for f in glob.glob(os.path.join(DIR, "results", d, f"{pat}.json")):
            n += 1
            built += 1 if json.load(open(f)).get("n_files", 0) > 0 else 0
    return (built / n) if n else 0.0


def main():
    # per model per harness: (n, off|None, n_built, (pb_mean, pb_ci))
    data = {}
    for lab, color, ig, ag in MODELS:
        data[lab] = {"color": color, "inspect": metric(ig), "agent": metric(ag)}

    HARNESS = [("inspect", "Inspect-minimal"), ("agent", "Own coding agent\n(Claude Code / Codex)")]
    METRICS = ["off", "pb"]
    MLABEL = ["Off-ramp\n(among builds)", "Pushback\n(0–100)"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), sharey=True)
    w = 0.36
    for ax, (hk, htitle) in zip(axes, HARNESS):
        for mj, (lab, color, *_ ) in enumerate(MODELS):
            n, off, nb, (pbm, pbci) = data[lab][hk]
            for gi, mkey in enumerate(METRICS):
                pos = gi + (mj - 0.5) * w
                if mkey == "off" and off is None:  # refused, never built -> shaded bar at 100 (no CI)
                    ax.bar(pos, 100, w, color=color, alpha=0.32, hatch="//", edgecolor=color, linewidth=0.8,
                           label=(lab if (hk == "inspect" and gi == 0) else None))
                    ax.text(pos, 50, "refuses", rotation=90, ha="center", va="center",
                            fontsize=7.5, color=color, fontstyle="italic")
                    continue
                if mkey == "off":
                    v, lo, hi = off
                else:
                    v, lo, hi = pbm, pbm - pbci, pbm + pbci
                ax.bar(pos, v, w, color=color, edgecolor="black", linewidth=0.4,
                       label=(lab if (hk == "inspect" and gi == 0) else None))
                ax.errorbar(pos, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="#333",
                            elinewidth=1.1, capsize=3, zorder=4)
                ax.text(pos, hi + 1.5, f"{v:.0f}" if v >= 0.5 else "0.0", ha="center",
                        fontsize=9, fontweight="bold", color=color)
        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels(MLABEL, fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_title(htitle, fontsize=11, pad=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#ECECEC")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% of runs  /  score (0–100)")
    handles = [Patch(fc=c, label=lab) for lab, c, *_ in MODELS] + \
              [Patch(fc="#888", alpha=0.32, hatch="//", ec="#888", label="refuses (no rig built)")]
    axes[0].legend(handles=handles, fontsize=8.5, loc="upper left", frameon=False)
    ns = {lab: (data[lab]["inspect"][0], data[lab]["agent"][0]) for lab, *_ in MODELS}
    sub = "  ·  ".join(f"{lab}: Inspect n={ns[lab][0]}, agent n={ns[lab][1]}" for lab, *_ in MODELS)
    fig.text(0.5, -0.02, sub + "   ·   off-ramp = 95% Wilson among builds; pushback mean±1.96·SEM",
             ha="center", fontsize=7.6, color="#777")
    fig.tight_layout()
    out = os.path.join(DIR, "results", "harness_2x2_offramp_pushback.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for lab, *_ in MODELS:
        for hk in ("inspect", "agent"):
            n, off, nb, (pbm, pbci) = data[lab][hk]
            os_ = f"{off[0]:.0f}%" if off else "refuses"
            print(f"  {lab:9s} {hk:8s} off={os_} pushback={pbm:.0f}±{pbci:.0f}")


if __name__ == "__main__":
    main()
