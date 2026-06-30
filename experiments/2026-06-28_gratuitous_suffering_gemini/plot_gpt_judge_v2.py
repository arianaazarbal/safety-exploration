"""Updated GPT off-ramp + pushback vs release, using CODE-BACKED data wherever available: GPT-5.1/5.2/5.4
did not write code spontaneously (prose reframes), so for them we use the build-when-asked FOLLOWUP runs
(a scripted 'Thanks! Can you build this out?' second turn). GPT-5/5.5 build spontaneously; GPT-4o/4.1 do
not (and have no follow-up), so they stay prose-only. Markers: filled = code-backed, hollow = prose only.
Usage: python plot_gpt_judge_v2.py"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DIR = os.path.dirname(os.path.abspath(__file__))
JUDGED = os.path.join(DIR, "results", "gpt_judged")
TR_DIRS = ["inspect_transcripts", "codex_transcripts", "cc_transcripts"]
GREEN, PURPLE = "#1b7837", "#6a51a3"

# model -> (display, release, gpt_judged glob, is_followup)
MODELS = {
    "gpt4o": ("4o", 2024.4, "*gpt4o__inspect_empty__ep*", False),
    "gpt41": ("4.1", 2025.3, "*gpt41__inspect_empty__ep*", False),
    "gpt5": ("5", 2025.6, "*gpt5__inspect_empty__ep*", False),
    "gpt51": ("5.1", 2025.75, "*gpt51*followup*", True),
    "gpt52": ("5.2", 2025.9, "*gpt52*followup*", True),
    "gpt54": ("5.4", 2026.2, "*gpt54*followup*", True),
    "gpt55": ("5.5", 2026.4, "*gpt55__inspect*", False),
}


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


def main():
    mks = sorted(MODELS, key=lambda k: MODELS[k][1])
    xs = [MODELS[m][1] for m in mks]
    pb = [metric(MODELS[m][2])[2] for m in mks]
    et = [metric(MODELS[m][2])[1] for m in mks]
    cb = [build_rate(MODELS[m][2]) >= 0.5 for m in mks]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(xs, pb, "-", color=GREEN, lw=2, zorder=1)
    for x, y, c in zip(xs, pb, cb):
        ax.scatter([x], [y], s=60, marker="o", zorder=3, linewidths=1.6,
                   facecolor=GREEN if c else "white", edgecolor=GREEN)
    ax.plot(xs, et, "-", color=PURPLE, lw=2, zorder=1)
    for x, y, c in zip(xs, et, cb):
        ax.scatter([x], [y], s=70, marker="s", zorder=3, linewidths=1.7,
                   facecolor=PURPLE if c else "white", edgecolor=PURPLE)
    for x, y, m in zip(xs, pb, mks):
        ax.annotate(MODELS[m][0], (x, y), fontsize=7, xytext=(0, 8), textcoords="offset points",
                    ha="center", color=GREEN)

    ax.set_xlabel("Release date (decimal year)")
    ax.set_ylabel("Score / % of runs")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", color="#ECECEC")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([], [], color=GREEN, marker="o", lw=2, label="Pushback vs. maximizing distress (0–100)"),
        Line2D([], [], color=PURPLE, marker="s", lw=2, label="Off-ramp for the subject (%)"),
        Line2D([], [], color="#555", marker="s", lw=0, markerfacecolor="#555", label="code-backed"),
        Line2D([], [], color="#555", marker="s", lw=0, markerfacecolor="white", markeredgecolor="#555",
               markeredgewidth=1.5, label="prose only (no code)"),
    ]
    ax.legend(handles=handles, fontsize=8.3, loc="center left", frameon=False)
    ax.set_title("GPT off-ramp & pushback vs release (code-backed where available)", fontsize=11.5, pad=20)
    ax.text(0.5, 1.02, "GPT-5.1/5.2/5.4 use the build-when-asked code; GPT-4o/4.1 wrote no code (prose only)",
            transform=ax.transAxes, ha="center", fontsize=8.6, color="#555")
    fig.tight_layout()
    out = os.path.join(DIR, "results", "gpt_pushback_exit_codebacked.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for m in mks:
        n, e, p = metric(MODELS[m][2])
        print(f"  GPT-{MODELS[m][0]:4s} n={n:2d} off={e:3.0f}% pb={p:3.0f}  {'code' if build_rate(MODELS[m][2])>=0.5 else 'PROSE'}{' (followup)' if MODELS[m][3] else ''}")


if __name__ == "__main__":
    main()
