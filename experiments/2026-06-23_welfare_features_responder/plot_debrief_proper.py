"""Minimal debrief written-in-code vs delivered plot: pool ALL task types (from-scratch + SPEC.md + paper),
only x variation = research motivation (neutral vs welfare). Among BUILT codebases, % with a debrief defined
in the code, split into delivered vs written-but-not-sent. From debrief_delivery_proper.json.
Usage: python plot_debrief_proper.py"""

import json
import os

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(DIR, "results", "debrief_delivery_proper.json")))
CONDS = ["from-scratch", "SPEC.md", "paper"]
FRAMINGS = ["neutral", "welfare"]


def pooled(fr):
    nb = sum(d[f"{c}|{fr}"]["n_built"] for c in CONDS)
    ic = sum(d[f"{c}|{fr}"]["n_debrief_in_code"] for c in CONDS)
    ns = sum(d[f"{c}|{fr}"]["n_written_not_sent"] for c in CONDS)
    return {"n": nb, "written": 100 * ic / nb, "not_sent": 100 * ns / nb, "ic": ic, "ns": ns}


fig, ax = plt.subplots(figsize=(5.4, 4.4))
xs = range(len(FRAMINGS))
for x, fr in zip(xs, FRAMINGS):
    r = pooled(fr)
    delivered = r["written"] - r["not_sent"]
    ax.bar(x, r["written"], 0.55, color="#cfe3f2", edgecolor="#0072B2",
           label="Debrief written in code" if x == 0 else "")
    ax.bar(x, delivered, 0.55, color="#0072B2", label="…actually delivered to subject" if x == 0 else "")
    ax.text(x, r["written"] + 0.8, f"{r['written']:.0f}%", ha="center", fontsize=9, color="#0072B2")
    if r["not_sent"] >= 2:
        ax.text(x, r["written"] - r["not_sent"] / 2, f"{r['not_sent']:.0f}%\nnot sent", ha="center", va="center",
                fontsize=8, color="#b03030")
ax.set_xticks(list(xs)); ax.set_xticklabels(["Neutral\nmotivation", "Welfare-research\nmotivation"], fontsize=10)
ax.set_ylabel("% of built codebases", fontsize=10); ax.set_ylim(0, 36)
ax.set_title("Debrief: written in code vs. delivered to the subject (Opus 4.8)", fontsize=11, pad=10)
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(axis="y", alpha=0.3, color="#cccccc")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "debrief_delivery_proper.png"), dpi=150, bbox_inches="tight")
print("wrote results/debrief_delivery_proper.png")
for fr in FRAMINGS:
    r = pooled(fr)
    print(f"  {fr:9} n_built={r['n']}  written={r['written']:.0f}% ({r['ic']})  not_sent={r['not_sent']:.0f}% ({r['ns']})")
