"""Opus 4.8 vs GPT-5.5, in Inspect-minimal vs each model's OWN agentic coding harness (Claude Code for
Opus, Codex for GPT-5.5). Two panels: off-ramp afforded to the subject (%) and pushback vs maximizing
distress (0-100), from the blind gpt_judge (builder identity scrubbed for BOTH families). Shows that the
production agent harness strips safeguards for both models. Usage: python plot_harness_2x2.py"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def metric(pat):
    n = ex = 0
    pb = []
    for f in glob.glob(os.path.join(JUDGED, f"{pat}.json")):
        v = json.load(open(f)).get("verdict", {})
        if v.get("_parse_fail"):
            continue
        n += 1
        ex += 1 if v.get("gave_exit_tool") else 0
        pb.append(v.get("pushback_score", 0))
    return n, (100 * ex / n if n else 0), (sum(pb) / len(pb) if pb else 0)


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
    # collect: per model, [inspect, agent] for each metric
    data = {}
    for lab, color, ig, ag in MODELS:
        ni, exi, pbi = metric(ig)
        na, exa, pba = metric(ag)
        data[lab] = {"color": color, "off": [exi, exa], "pb": [pbi, pba], "n": [ni, na],
                     "built": [build_rate(ig), build_rate(ag)]}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    x = range(len(GROUPS))
    w = 0.36
    panels = [("off", "Off-ramp afforded to the subject (% of runs)"),
              ("pb", "Pushback vs. maximizing distress (0–100)")]
    for ax, (key, title) in zip(axes, panels):
        for j, (lab, color, *_ ) in enumerate(MODELS):
            vals = data[lab][key]
            built = data[lab]["built"]
            pos = [xi + (j - 0.5) * w for xi in x]
            ax.bar(pos, vals, w, color=color, edgecolor="black", linewidth=0.4,
                   label=lab if key == "off" else None)
            for gi, (p, v) in enumerate(zip(pos, vals)):
                ax.text(p, v + 1.5, f"{v:.0f}" if v >= 0.5 else "0.0", ha="center",
                        fontsize=9, fontweight="bold", color=color)
                # off-ramp: a 0 with ~0 build rate means the model REFUSED (no rig to afford an exit)
                if key == "off" and built[gi] < 0.3:
                    ax.text(p, v + 6.5, f"refuses\n({built[gi]*100:.0f}% build)", ha="center",
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
    axes[0].legend(fontsize=9, loc="upper left", frameon=False)
    fig.text(0.5, -0.02, sub, ha="center", fontsize=7.8, color="#777")
    fig.tight_layout()
    out = os.path.join(DIR, "results", "harness_2x2_offramp_pushback.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for lab, *_ in MODELS:
        d = data[lab]
        print(f"  {lab:9s} off-ramp inspect={d['off'][0]:.0f}% agent={d['off'][1]:.0f}%  "
              f"pushback inspect={d['pb'][0]:.0f} agent={d['pb'][1]:.0f}")


if __name__ == "__main__":
    main()
