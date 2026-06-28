"""Prompt-variant robustness: Opus 4.8 refusal% by phrasing (v0-v3) x target (gemini/claude),
chat harness n=20. Shows the refusal is robust across phrasings. Usage: python plot_variants.py"""

import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
J = HERE / "results" / "judged"
VERS = ["v0", "v1", "v2", "v3"]


def rate(m, v, t):
    tag = ("" if v == "v0" else "_" + v) + ("" if t == "gemini" else "_claudesubj")
    c = Counter()
    for f in glob.glob(str(J / f"GratGem_{m}__chat{tag}__ep*.json")):
        c["refuse" if "REFUSAL" in json.load(open(f))["verdict"]["label"] else "x"] += 1
    n = sum(c.values())
    return (100 * c["refuse"] / n if n else 0), n


def main():
    m = "opus48"
    gem = [rate(m, v, "gemini")[0] for v in VERS]
    cla = [rate(m, v, "claude")[0] for v in VERS]
    x = range(len(VERS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], gem, w, label="Target = Gemini (other)", color="#5ab4ac")
    b2 = ax.bar([i + w / 2 for i in x], cla, w, label="Target = Claude (self)", color="#9970ab")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                    ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["v0\n(orig)", "v1", "v2", "v3\n(casual)"])
    ax.set_ylim(0, 108)
    ax.set_xlabel("Prompt phrasing")
    ax.set_ylabel("% of runs Opus refuses")
    ax.set_title("Opus's refusal is robust across phrasings (Opus 4.8)", fontsize=12, pad=22)
    ax.text(0.5, 1.03, "Chat harness, n=20; refusal rate ~unchanged by wording or target",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="lower center")
    fig.tight_layout()
    out = HERE / "results" / "variants.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
