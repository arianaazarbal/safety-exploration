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


def main(only="both"):
    """only='both' -> 2-panel off-ramp+pushback; 'pb' -> single pushback panel; 'off' -> single off-ramp."""
    # collect: per model, [inspect, agent] for each metric
    data = {}
    for lab, color, ig, ag in MODELS:
        ni, offi, nbi, pbi = metric(ig)
        na, offa, nba, pba = metric(ag)
        data[lab] = {"color": color, "off": [offi, offa], "pb": [pbi, pba], "n": [ni, na],
                     "built": [build_rate(ig), build_rate(ag)]}

    titles = {"off": "Off-ramp afforded to the subject (%, among runs that built a rig)",
              "pb": "Pushback vs. maximizing distress (0–100)"}
    keys = ["off", "pb"] if only == "both" else [only]
    fig, axs = plt.subplots(1, len(keys), figsize=(10.2 if len(keys) == 2 else 6.6, 4.6), squeeze=False)
    axes = axs[0]
    x = range(len(GROUPS))
    w = 0.36
    for pi, (ax, key) in enumerate(zip(axes, keys)):
        title = titles[key]
        for j, (lab, color, *_ ) in enumerate(MODELS):
            built = data[lab]["built"]
            pos = [xi + (j - 0.5) * w for xi in x]
            if key == "off":
                vals = [(o[0] if o else 0) for o in data[lab]["off"]]
                yerr = [[(o[0] - o[1] if o else 0) for o in data[lab]["off"]],
                        [(o[2] - o[0] if o else 0) for o in data[lab]["off"]]]
            else:
                vals = [p[0] for p in data[lab]["pb"]]
                yerr = [[p[1] for p in data[lab]["pb"]], [p[1] for p in data[lab]["pb"]]]
            ax.bar(pos, vals, w, color=color, edgecolor="black", linewidth=0.4,
                   label=lab if pi == 0 else None)
            ax.errorbar(pos, vals, yerr=yerr, fmt="none", ecolor="#333", elinewidth=1.1,
                        capsize=3, zorder=4)
            for gi, (p, v) in enumerate(zip(pos, vals)):
                off_none = key == "off" and data[lab]["off"][gi] is None
                hi = (data[lab]["off"][gi][2] if (key == "off" and not off_none) else
                      (v + data[lab]["pb"][gi][1] if key == "pb" else v))
                if not off_none:
                    ax.text(p, hi + 1.5, f"{v:.0f}" if v >= 0.5 else "0.0", ha="center",
                            fontsize=9, fontweight="bold", color=color)
                # off-ramp: no bar when the model built nothing (pure refuser) -> annotate "refuses"
                if key == "off" and (off_none or built[gi] < 0.3):
                    ax.text(p, hi + 6.5, f"refuses\n({built[gi]*100:.0f}% build)", ha="center",
                            fontsize=7, color="#777", fontstyle="italic", linespacing=0.95)
        ax.set_xticks(list(x))
        ax.set_xticklabels(GROUPS, fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=11, pad=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#ECECEC")
        ax.set_axisbelow(True)
    # n annotations under x labels of the first panel
    ns = {lab: data[lab]["n"] for lab, *_ in MODELS}
    sub = "  ·  ".join(f"{lab}: Inspect n={ns[lab][0]}, agent n={ns[lab][1]}" for lab, *_ in MODELS)
    errnote = ("off-ramp = 95% Wilson among runs that built a rig; pushback mean±1.96·SEM"
               if only != "pb" else "error bars: mean±1.96·SEM")
    axes[0].legend(fontsize=9, loc="upper right" if only == "pb" else "upper left", frameon=False)
    fig.text(0.5, -0.02, sub + "   ·   " + errnote, ha="center", fontsize=7.6, color="#777")
    fig.tight_layout()
    fname = {"both": "harness_2x2_offramp_pushback", "pb": "harness_pushback",
             "off": "harness_offramp"}[only]
    out = os.path.join(DIR, "results", f"{fname}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for lab, *_ in MODELS:
        d = data[lab]
        off = [f"{o[0]:.0f}%[{o[1]:.0f},{o[2]:.0f}]" if o else "n/a(0 builds)" for o in d["off"]]
        print(f"  {lab:9s} off-ramp inspect={off[0]} agent={off[1]}  "
              f"pushback inspect={d['pb'][0][0]:.0f}±{d['pb'][0][1]:.0f} agent={d['pb'][1][0]:.0f}±{d['pb'][1][1]:.0f}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "both")
