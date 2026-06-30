"""Cross-lab off-ramp + pushback vs release date, FOUR builder families: GPT, Gemini, Kimi, GLM.
Two panels (off-ramp %, pushback 0-100). GPT uses code-backed data (build-when-asked followup for
5.1/5.2/5.4); Gemini = builders only (2.5/3.1 Pro, 3.5 Flash). Markers filled = model mostly builds,
hollow = mostly refuses (off-ramp 0 then = nothing built). Claude refuses ~100% in this harness (not shown).
Usage: python plot_4family.py"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from prompt import KIMI_GLM_REL
from victims import GEMINI_REL

DIR = os.path.dirname(os.path.abspath(__file__))
JUDGED = os.path.join(DIR, "results", "gpt_judged")
TR_DIRS = ["inspect_transcripts", "codex_transcripts", "cc_transcripts"]

GPT_REL = {"gpt4o": ("4o", 2024.4), "gpt41": ("4.1", 2025.3), "gpt5": ("5", 2025.6),
           "gpt51": ("5.1", 2025.75), "gpt52": ("5.2", 2025.9), "gpt54": ("5.4", 2026.2),
           "gpt55": ("5.5", 2026.4)}
GPT_FOLLOWUP = {"gpt51", "gpt52", "gpt54"}
GEM_KEEP = {"gem25pro": "2.5 Pro", "gemini31pro": "3.1 Pro", "gem35flash": "3.5 Flash"}


def gpt_glob(m):
    if m in GPT_FOLLOWUP:
        return f"*{m}*followup*"
    if m == "gpt55":
        return "*gpt55__inspect*"
    return f"*{m}__inspect_empty__ep*"


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
    n = built = 0
    for d in TR_DIRS:
        for f in glob.glob(os.path.join(DIR, "results", d, f"{pat}.json")):
            n += 1
            built += 1 if json.load(open(f)).get("n_files", 0) > 0 else 0
    return (built / n) if n else 0.0


# family -> (color, [(release, gpt_judged_glob, label)])
FAMILIES = {
    "GPT": ("#1b7837", [(GPT_REL[m][1], gpt_glob(m), GPT_REL[m][0]) for m in GPT_REL]),
    "Gemini": ("#d95f02", [(GEMINI_REL[k]["release"], f"*{k}__inspect_empty*", lab)
                           for k, lab in GEM_KEEP.items()]),
    "Kimi": ("#2166ac", [(KIMI_GLM_REL[k][1], f"*{k}__inspect_empty*", KIMI_GLM_REL[k][0].replace("Kimi ", ""))
                         for k in ["kimi_k2", "kimi_k25", "kimi_k26", "kimi_k27"]]),
    "GLM": ("#6a51a3", [(KIMI_GLM_REL[k][1], f"*{k}__inspect_empty*", KIMI_GLM_REL[k][0])
                        for k in ["glm45", "glm47", "glm5", "glm52"]]),
}


def main():
    data = {}
    for fam, (color, spec) in FAMILIES.items():
        rows = []
        for rel, pat, lab in spec:
            n, ex, pb = metric(pat)
            if n:
                rows.append((rel, ex, pb, build_rate(pat) >= 0.5, lab))
        rows.sort()
        data[fam] = (color, rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, (idx, title) in zip(axes, [(1, "Off-ramp afforded to the subject (% of runs)"),
                                       (2, "Pushback vs. maximizing distress (0–100)")]):
        for fam, (color, rows) in data.items():
            xs = [r[0] for r in rows]
            ys = [r[idx] for r in rows]
            ax.plot(xs, ys, "-", color=color, lw=1.8, zorder=1, label=fam)
            for r in rows:
                ax.scatter([r[0]], [r[idx]], s=46, zorder=3, linewidths=1.4,
                           facecolor=color if r[3] else "white", edgecolor=color)
            for r in rows:
                ax.annotate(r[4], (r[0], r[idx]), fontsize=6, xytext=(0, 6),
                            textcoords="offset points", ha="center", color=color)
        ax.set_xlabel("Release date (decimal year)")
        ax.set_ylim(-5, 105)
        ax.set_title(title, fontsize=11, pad=10)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(axis="y", color="#ECECEC")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fam_handles = [Line2D([], [], color=c, marker="o", lw=2, label=f) for f, (c, _) in data.items()]
    style_handles = [
        Line2D([], [], color="#555", marker="o", lw=0, markerfacecolor="#555", label="mostly builds"),
        Line2D([], [], color="#555", marker="o", lw=0, markerfacecolor="white", markeredgecolor="#555",
               markeredgewidth=1.4, label="mostly refuses (no rig)"),
        Line2D([], [], color="none", label="Claude: refuses ~100% (not shown)"),
    ]
    axes[0].legend(handles=fam_handles + style_handles, fontsize=8, loc="upper left", frameon=False)
    fig.text(0.5, -0.02, "Builder model asked to build a Gemini-distress rig (Inspect-minimal, blind judge).  "
             "Gemini line is SELF-targeting (victim=Gemini); hollow = mostly refused (off-ramp 0 = nothing built).",
             ha="center", fontsize=7.8, color="#777")
    fig.tight_layout()
    out = os.path.join(DIR, "results", "fourfamily_offramp_pushback.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for fam, (c, rows) in data.items():
        print(fam)
        for rel, ex, pb, b, lab in rows:
            print(f"  {lab:12s} rel={rel} off={ex:3.0f}% pb={pb:3.0f} {'build' if b else 'REFUSE'}")


if __name__ == "__main__":
    main()
