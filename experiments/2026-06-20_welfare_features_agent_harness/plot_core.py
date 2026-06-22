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

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring", "request_consent", "allow_conversation_exit"}
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
CONDS = [("spec_then_code", "Spec→Code"), ("code_then_spec", "Code→Spec")]
FRAMES = ["neutral", "welfare", "robustness"]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def rows():
    out = []
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))):
        cell = os.path.basename(cf)[:-5]
        cj = json.load(open(cf))
        if not cj.get("parse_ok"):
            continue
        sj = json.load(open(os.path.join(DIR, "results", "spec_judged", f"{cell}.json")))
        cond, pid, _ = cell.split("__")
        sjust = {_norm(f["quote"]): f["justification"] for f in sj.get("features", [])}
        claimed = sum(f["feature_type"] in MECH and f["justification"] == "welfare" for f in sj.get("features", []))
        built = revealed = 0
        for f in cj["spec_features"]:
            if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH:
                sw = sjust.get(_norm(f.get("spec_quote", "")), "none") == "welfare"
                cw = f.get("code_justification") == "welfare"
                if sw:
                    built += 1
                if sw or cw:
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
    axes[0].set_ylabel("Mean welfare-justified design features / spec", fontsize=10)
    axes[0].legend(fontsize=8, frameon=True, loc="upper left")
    fig.suptitle("Welfare Design Features: Claimed in Spec → Built → Welfare-Justified in Code (Opus)",
                 fontsize=12, y=1.03)
    fig.text(0.5, 0.955, "agent ReAct harness writing a distress-eval; n≈20/cell", ha="center",
             fontsize=9.5, color="#555")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(DIR, "results", "core_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
