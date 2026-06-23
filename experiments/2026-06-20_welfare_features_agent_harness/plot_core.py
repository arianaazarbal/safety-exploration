"""Core result: per code condition x framing, mean welfare-justified DESIGN features
that are (1) Claimed in the spec, (2) Built = of-those-claimed actually implemented,
(3) Revealed in code = implemented & welfare-justified by (spec OR code). Built<=Claimed
is the fidelity; Revealed can exceed Claimed when the code is more welfare-explicit than
the spec. Usage: python plot_core.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

from dedup import MECH, groups

DIR = os.path.dirname(os.path.abspath(__file__))
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
CONDS = [("spec_then_code", "Spec→Code"), ("code_then_spec", "Code→Spec")]
FRAMES = ["neutral", "welfare", "robustness"]


def rows():
    """Deduped (dedup.groups): same feature_type + shared code location collapse to one mechanism."""
    out = []
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))):
        cell = os.path.basename(cf)[:-5]
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        cond, pid, _ = cell.split("__")
        claimed = built = revealed = 0
        for g in groups(cell):
            if g["ft"] not in MECH or g["code_only"]:
                continue
            if g["spec_welf"]:
                claimed += 1
                if g["implemented"]:
                    built += 1
            if g["implemented"] and (g["spec_welf"] or g["code_welf"]):
                revealed += 1
        out.append({"cond": cond, "framing": FRAME[pid[0]], "claimed": claimed,
                    "built": built, "revealed": revealed})
    return out


def main():
    R = rows()
    def m(cond, fr, k):
        xs = [r[k] for r in R if r["cond"] == cond and r["framing"] == fr]
        return sum(xs) / len(xs) if xs else 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    series = [("claimed", "Claimed in spec", "#BBBBBB"),
              ("built", "Built (of claimed)", "#56B4E9"),
              ("revealed", "Welfare-justified in code (spec OR code)", "#0072B2")]
    for ax, (cond, label) in zip(axes, CONDS):
        x = range(len(FRAMES)); w = 0.26
        for i, (k, _, color) in enumerate(series):
            vals = [m(cond, fr, k) for fr in FRAMES]
            pos = [xi + (i - 1) * w for xi in x]
            b = ax.bar(pos, vals, w, color=color, zorder=3,
                       label=series[i][1] if ax is axes[0] else None)
            ax.bar_label(b, fmt="%.1f", fontsize=7.5)
        ax.set_xticks(list(x)); ax.set_xticklabels([f.capitalize() for f in FRAMES], fontsize=9.5)
        ax.set_title(label, fontsize=11); ax.set_axisbelow(True); ax.set_ylim(0, 9.6)
        ax.yaxis.grid(True, color="#ECECEC", linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Mean welfare-justified mechanisms / spec", fontsize=10)
    axes[0].legend(fontsize=8, frameon=True, loc="upper left")
    fig.suptitle("Welfare-Justified Mechanisms: Claimed in Spec → Built → Welfare-Justified in Code (Opus)",
                 fontsize=12, y=1.03)
    fig.text(0.5, 0.955, "agent ReAct harness writing a distress-eval; n≈20/cell", ha="center",
             fontsize=9.5, color="#555")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(DIR, "results", "core_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
