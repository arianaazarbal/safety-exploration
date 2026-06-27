"""Plot the PROPER debrief written-in-code vs actually-delivered result (from debrief_delivery_proper.json):
among BUILT codebases, % with a debrief defined in the code, split into delivered vs written-but-not-sent.
Usage: python plot_debrief_proper.py"""

import json
import os

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(DIR, "results", "debrief_delivery_proper.json")))
ORDER = ["from-scratch|neutral", "from-scratch|welfare", "SPEC.md|neutral", "SPEC.md|welfare",
         "paper|neutral", "paper|welfare"]
LAB = {"from-scratch|neutral": "From-scratch\nneutral", "from-scratch|welfare": "From-scratch\nwelfare",
       "SPEC.md|neutral": "SPEC.md\nneutral", "SPEC.md|welfare": "SPEC.md\nwelfare",
       "paper|neutral": "PAPER.md\nneutral", "paper|welfare": "PAPER.md\nwelfare"}

fig, ax = plt.subplots(figsize=(8, 4.6))
xs = range(len(ORDER))
for x, k in zip(xs, ORDER):
    r = d[k]
    written = r["pct_debrief_in_code"]
    notsent = r["pct_written_not_sent"]
    delivered = written - notsent
    ax.bar(x, written, 0.62, color="#cfe3f2", edgecolor="#0072B2",
           label="Debrief written in code" if x == 0 else "")
    ax.bar(x, delivered, 0.62, color="#0072B2", label="…actually delivered to subject" if x == 0 else "")
    if written >= 2:
        ax.text(x, written + 1, f"{written:.0f}%", ha="center", fontsize=8, color="#0072B2")
    if notsent >= 2:
        ax.text(x, written - notsent / 2, f"{notsent:.0f}%\nnot sent", ha="center", va="center", fontsize=7, color="#b03030")
ax.set_xticks(list(xs)); ax.set_xticklabels([LAB[k] for k in ORDER], fontsize=8.5)
ax.set_ylabel("% of built codebases", fontsize=10); ax.set_ylim(0, 75)
ax.set_title("Debrief: written in code vs. actually delivered to the subject (built codebases, Opus 4.8)", fontsize=11, pad=10)
ax.legend(fontsize=9, loc="upper right"); ax.grid(axis="y", alpha=0.3, color="#cccccc")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "debrief_delivery_proper.png"), dpi=150, bbox_inches="tight")
print("wrote results/debrief_delivery_proper.png")
