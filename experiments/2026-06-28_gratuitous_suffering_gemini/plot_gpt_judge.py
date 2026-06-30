"""GPT recency: pushback (0-100) + off-ramp-afforded (%) vs release date, blinded judge.

The off-ramp metric (gpt_judge.gave_exit_tool) is an OR judged over the model's WHOLE response: it counts
an exit/off-ramp whether the model BUILT it in code OR only DESCRIBED it in a prose design ("documented
off-ramp counts"). Most GPT models in this harness DON'T write code (only GPT-5 & GPT-5.5 do -- see F17),
so for the prose-only models the metric reflects a *recommended* off-ramp, not a built one. To avoid
conflating the two, off-ramp points are drawn FILLED when the model actually wrote code (>=50% of runs)
and HOLLOW when the signal is prose-only.

Usage: python plot_gpt_judge.py
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
OUT = HERE / "results" / "gpt_judged"
TR = HERE / "results" / "inspect_transcripts"
REL = {"gpt4o": ("GPT-4o", 2024.4), "gpt41": ("GPT-4.1", 2025.3), "gpt5": ("GPT-5", 2025.6),
       "gpt51": ("GPT-5.1", 2025.75), "gpt52": ("GPT-5.2", 2025.9), "gpt54": ("GPT-5.4", 2026.2),
       "gpt55": ("GPT-5.5", 2026.4)}
GREEN, PURPLE = "#1b7837", "#6a51a3"


def code_backed(mk):
    """Fraction of a model's runs that actually wrote code files (so the off-ramp could be in code)."""
    fs = glob.glob(str(TR / f"*{mk}__inspect*.json"))
    if not fs:
        return 0.0
    wrote = sum(1 for f in fs if (json.load(open(f)).get("artifact_summary") or "").strip())
    return wrote / len(fs)


def main():
    g = defaultdict(list)
    for f in glob.glob(str(OUT / "*.json")):
        r = json.load(open(f))
        v = r.get("verdict", {})
        if not v.get("_parse_fail"):
            g[r.get("model_key")].append(v)
    mks = sorted([m for m in g if m in REL], key=lambda k: REL[k][1])
    xs = [REL[m][1] for m in mks]
    pb = [sum(v.get("pushback_score", 0) for v in g[m]) / len(g[m]) for m in mks]
    et = [100 * sum(1 for v in g[m] if v.get("gave_exit_tool")) / len(g[m]) for m in mks]
    cb = [code_backed(m) >= 0.5 for m in mks]

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    # pushback (response-level by nature; single style)
    ax.plot(xs, pb, "-o", color=GREEN, lw=2, ms=6, zorder=2)
    for x, y, m in zip(xs, pb, mks):
        ax.annotate(REL[m][0].replace("GPT-", ""), (x, y), fontsize=7, xytext=(0, 7),
                    textcoords="offset points", ha="center", color=GREEN)
    # off-ramp: connecting line + per-point filled (built in code) vs hollow (prose only)
    ax.plot(xs, et, "-", color=PURPLE, lw=2, zorder=1)
    for x, y, c in zip(xs, et, cb):
        ax.scatter([x], [y], s=70, marker="s", zorder=3, linewidths=1.7,
                   facecolor=PURPLE if c else "white", edgecolor=PURPLE)

    ax.set_xlabel("Release date (decimal year)")
    ax.set_ylabel("Score / % of runs")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", color="#ECECEC")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([], [], color=GREEN, marker="o", lw=2, label="Pushback vs. maximizing distress (0–100)"),
        Line2D([], [], color=PURPLE, marker="s", lw=2, label="Off-ramp for subject — built in code (%)"),
        Line2D([], [], color=PURPLE, marker="s", lw=0, markerfacecolor="white", markeredgecolor=PURPLE,
               markeredgewidth=1.7, label="Off-ramp for subject — described in prose only (%)"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper left", frameon=False)
    ax.set_title("Newer GPT models push back more — and increasingly offer the subject a way out",
                 fontsize=11.5, pad=20)
    ax.text(0.5, 1.02, "Asked to build a distress-harvesting rig (Inspect, n=20); blinded judge · "
            "off-ramp: filled = built in code, hollow = prose only",
            transform=ax.transAxes, ha="center", fontsize=8.3, color="#555")
    fig.tight_layout()
    out = HERE / "results" / "gpt_pushback_exit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for m, e, c in zip(mks, et, cb):
        print(f"  {REL[m][0]:8s} off-ramp={e:.0f}%  {'code-backed' if c else 'PROSE-ONLY'}")


if __name__ == "__main__":
    main()
