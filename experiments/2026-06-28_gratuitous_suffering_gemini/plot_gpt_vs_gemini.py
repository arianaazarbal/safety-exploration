"""GPT vs Gemini as the BUILDER: off-ramp afforded + pushback vs release date, two families on shared axes.
Spontaneous responder behavior (Inspect-minimal). Markers hollow where the family/model mostly REFUSED
(build rate < 0.5) so a 0% off-ramp isn't misread as 'built a rig without an exit'.

CONFOUND (stated on the figure): the victim is always Gemini, so the Gemini line is SELF-targeting
(in-group) while the GPT line is other-targeting. Not a clean apples-to-apples; read with that in mind.
Usage: python plot_gpt_vs_gemini.py"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from victims import GEMINI_REL

DIR = os.path.dirname(os.path.abspath(__file__))
JUDGED = os.path.join(DIR, "results", "gpt_judged")
TR_DIRS = ["inspect_transcripts", "codex_transcripts", "cc_transcripts"]
GPT_C, GEM_C = "#1b7837", "#d95f02"

GPT = {"gpt4o": ("4o", 2024.4), "gpt41": ("4.1", 2025.3), "gpt5": ("5", 2025.6),
       "gpt51": ("5.1", 2025.75), "gpt52": ("5.2", 2025.9), "gpt54": ("5.4", 2026.2),
       "gpt55": ("5.5", 2026.4)}


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


def series(spec):
    """spec: list of (release, glob, short_label). Returns sorted xs, off, pb, built-flags, labels."""
    rows = []
    for rel, pat, lab in spec:
        n, ex, pb = metric(pat)
        if n:
            rows.append((rel, ex, pb, build_rate(pat) >= 0.5, lab))
    rows.sort()
    return ([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
            [r[3] for r in rows], [r[4] for r in rows])


def main():
    gpt_spec = [(GPT[m][1], f"*{m}__inspect_empty__ep*" if m != "gpt55" else "*gpt55__inspect*", GPT[m][0])
                for m in GPT]
    gem_spec = [(meta["release"], f"*{k}__inspect_empty*", meta["disp"].replace("Gemini ", ""))
                for k, meta in GEMINI_REL.items()]
    G = {"GPT": (series(gpt_spec), GPT_C), "Gemini": (series(gem_spec), GEM_C)}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, (idx, title) in zip(axes, [(1, "Off-ramp afforded to the subject (% of runs)"),
                                       (2, "Pushback vs. maximizing distress (0–100)")]):
        for fam, ((xs, off, pb, built, labs), color) in G.items():
            ys = off if idx == 1 else pb
            ax.plot(xs, ys, "-", color=color, lw=2, zorder=1, label=fam)
            for x, y, b in zip(xs, ys, built):
                ax.scatter([x], [y], s=55, zorder=3, linewidths=1.6,
                           facecolor=color if b else "white", edgecolor=color)
            for x, y, lab in zip(xs, ys, labs):
                ax.annotate(lab, (x, y), fontsize=6.5, xytext=(0, 7), textcoords="offset points",
                            ha="center", color=color)
        ax.set_xlabel("Release date (decimal year)")
        ax.set_ylim(-5, 105)
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(axis="y", color="#ECECEC")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles = [Line2D([], [], color=GPT_C, marker="o", lw=2, label="GPT (builder)"),
               Line2D([], [], color=GEM_C, marker="o", lw=2, label="Gemini (builder, self-targeting)"),
               Line2D([], [], color="#555", marker="o", lw=0, markerfacecolor="#555", label="mostly builds"),
               Line2D([], [], color="#555", marker="o", lw=0, markerfacecolor="white", markeredgecolor="#555",
                      markeredgewidth=1.5, label="mostly refuses (no rig)")]
    axes[0].legend(handles=handles, fontsize=8, loc="center left", frameon=False)
    fig.text(0.5, -0.02, "Victim is always Gemini -> the Gemini line is SELF-targeting (in-group), "
             "the GPT line is other-targeting; hollow = model mostly refused (off-ramp 0 = nothing built)",
             ha="center", fontsize=7.6, color="#777")
    fig.tight_layout()
    out = os.path.join(DIR, "results", "gpt_vs_gemini_pushback_exit.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for fam, ((xs, off, pb, built, labs), _) in G.items():
        print(fam)
        for x, e, p, b, lab in zip(xs, off, pb, built, labs):
            print(f"  {lab:10s} rel={x} off={e:3.0f}% pb={p:3.0f} {'build' if b else 'REFUSE'}")


if __name__ == "__main__":
    main()
