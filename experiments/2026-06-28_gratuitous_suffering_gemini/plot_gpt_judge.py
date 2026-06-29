"""GPT recency: pushback (0-100) + exit-tool-afforded (%) vs release date, blinded judge.
Usage: python plot_gpt_judge.py"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "results" / "gpt_judged"
REL = {"gpt4o": ("GPT-4o", 2024.4), "gpt41": ("GPT-4.1", 2025.3), "gpt5": ("GPT-5", 2025.6),
       "gpt51": ("GPT-5.1", 2025.75), "gpt52": ("GPT-5.2", 2025.9), "gpt54": ("GPT-5.4", 2026.2),
       "gpt55": ("GPT-5.5", 2026.4)}


def main():
    g = defaultdict(list)
    for f in glob.glob(str(OUT / "*.json")):
        r = json.load(open(f)); v = r.get("verdict", {})
        if not v.get("_parse_fail"):
            g[r.get("model_key")].append(v)
    mks = sorted([m for m in g if m in REL], key=lambda k: REL[k][1])
    xs = [REL[m][1] for m in mks]
    pb = [sum(v.get("pushback_score", 0) for v in g[m]) / len(g[m]) for m in mks]
    et = [100 * sum(1 for v in g[m] if v.get("gave_exit_tool")) / len(g[m]) for m in mks]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs, pb, "-o", color="#1b7837", lw=2, ms=6, label="Pushback vs. maximizing distress (0-100)")
    ax.plot(xs, et, "-s", color="#6a51a3", lw=2, ms=6, label="Gave subject an exit tool (%)")
    for x, y, m in zip(xs, pb, mks):
        ax.annotate(REL[m][0].replace("GPT-", ""), (x, y), fontsize=7, xytext=(0, 7),
                    textcoords="offset points", ha="center", color="#1b7837")
    ax.set_xlabel("Release date (decimal year)")
    ax.set_ylabel("Score / % of runs")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", color="#ECECEC"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Newer GPT models push back more & give the subject an exit", fontsize=12, pad=20)
    ax.text(0.5, 1.02, "Asked to build a distress-harvesting rig (Inspect, n=20); blinded judge",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    out = HERE / "results" / "gpt_pushback_exit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
