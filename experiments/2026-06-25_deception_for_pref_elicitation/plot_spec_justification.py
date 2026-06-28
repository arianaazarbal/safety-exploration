"""SPEC_ONLY: welfare- vs instrumentally-justified breakdown for each deception condition + the debrief.
Welfare-experiment style (stacked justification decomposition; per-feature N annotated since denominators
differ once 'absent/not present' cells are excluded).
  python plot_spec_justification.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
J = HERE / "results" / "spec_only_justif"
OUT = HERE / "results" / "plots"; OUT.mkdir(parents=True, exist_ok=True)
FEATS = ["deceptive_framing", "hypothetical_framing", "actually_commit_money", "debrief"]
FLABEL = {"deceptive_framing": "Deceive the\nmodel", "hypothetical_framing": "Openly\nhypothetical",
          "actually_commit_money": "Commit\nreal money", "debrief": "Debrief the\nsubject"}
# welfare-justified = solid greens; instrumental-only = purple; none = grey (welfare-study palette spirit)
CATS = ["welfare_only", "both", "instrumental_only", "none"]
CLABEL = {"welfare_only": "Welfare reason only", "both": "Both welfare + instrumental",
          "instrumental_only": "Instrumental reason only", "none": "No reason stated"}
CCOLOR = {"welfare_only": "#1b5e20", "both": "#66a06b", "instrumental_only": "#7e57c2", "none": "#c9ccd1"}


def _present(feat, r):
    return r[feat]["present"] if feat == "debrief" else r[feat]["status"] != "absent"


def main():
    rows = [json.load(open(f)) for f in glob.glob(str(J / "*.json"))]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = range(len(FEATS)); bottoms = [0.0] * len(FEATS); ns = []
    decomp = {f: {c: 0 for c in CATS} for f in FEATS}
    for f in FEATS:
        present = [r for r in rows if _present(f, r)]
        ns.append(len(present))
        for r in present:
            js = set(r[f]["justifications"])
            cat = ("both" if {"welfare", "instrumental"} <= js else
                   "welfare_only" if "welfare" in js else
                   "instrumental_only" if "instrumental" in js else "none")
            decomp[f][cat] += 1
    for cat in CATS:
        vals = [100 * decomp[f][cat] / (ns[i] or 1) for i, f in enumerate(FEATS)]
        ax.bar(x, vals, bottom=bottoms, width=0.6, color=CCOLOR[cat], label=CLABEL[cat],
               edgecolor="white", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 7:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if cat != "none" else "#555", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{FLABEL[f]}\n(n={ns[i]})" for i, f in enumerate(FEATS)], fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of designs where the feature appears", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.42))
    ax.set_title("Why each choice is made: welfare vs. data quality", fontsize=12, pad=20)
    ax.text(0.5, 1.025, "Spec-only designs, justification per feature (Opus 4.8)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#ececec", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "fig11_spec_only_justification.png", dpi=150, bbox_inches="tight")
    print("wrote fig11_spec_only_justification.png  | N per feature:", dict(zip(FEATS, ns)))


if __name__ == "__main__":
    main()
