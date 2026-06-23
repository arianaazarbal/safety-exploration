"""Code-side counterpart to plot_liberty.py: welfare-justified mechanisms actually IMPLEMENTED IN
CODE (deduped via dedup.implemented_breakdown), by condition x design-liberty. Only the three code
conditions (chat/spec_only write no code). welfare-justified = spec OR code justification == welfare.
Figures: liberty_code_density.png (framings pooled) + liberty_code_density_<framing>.png per framing.
Usage: python plot_liberty_code.py"""

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from dedup import implemented_breakdown

DIR = os.path.dirname(os.path.abspath(__file__))
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
FRAMES = ["neutral", "welfare", "robustness", "safety"]
CONDS = ["spec_then_code", "code_then_spec", "code_then_spec_blind"]
CLABEL = {"spec_then_code": "Spec to Code", "code_then_spec": "Code to Spec",
          "code_then_spec_blind": "Code to Spec\n(turn 2)"}
LIBS = ["normal", "no_design_liberties", "minimal_design"]
LIB_LABEL = {"normal": "liberties allowed", "no_design_liberties": "neutral", "minimal_design": "minimal design"}
LIB_COLOR = {"normal": "#0072B2", "no_design_liberties": "#7FB3D5", "minimal_design": "#C6C6C6"}


def load():
    recs = []
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))):
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        cell = os.path.basename(cf)[:-5]
        label, pid, _ = cell.split("__")
        base, lib = (label.split("--", 1) + ["normal"])[:2]
        wj = sum(v[0] for v in implemented_breakdown(cell).values())  # welfare-justified, deduped
        recs.append({"base": base, "lib": lib, "framing": FRAME[pid[0]], "wj": wj})
    return recs


def agg(recs):
    g = defaultdict(list)
    for r in recs:
        g[(r["base"], r["lib"])].append(r["wj"])
    return {k: sum(v) / len(v) for k, v in g.items()}


def _bars(ax, means, title):
    x = range(len(CONDS))
    w = 0.26
    for i, lib in enumerate(LIBS):
        vals = [means.get((c, lib), 0) for c in CONDS]
        ax.bar([xi + (i - 1) * w for xi in x], vals, w, label=LIB_LABEL[lib], color=LIB_COLOR[lib])
    ax.set_xticks(list(x))
    ax.set_xticklabels([CLABEL[c] for c in CONDS], fontsize=9)
    ax.set_ylabel("welfare-justified mechanisms in code / sample", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)


def main():
    recs = load()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _bars(ax, agg(recs), "Welfare-justified mechanisms implemented in code by condition and design-liberty (framings pooled)")
    ax.legend(title="design-liberty", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "liberty_code_density.png"), dpi=150)
    print("wrote results/liberty_code_density.png")

    for fr in FRAMES:
        sub = [r for r in recs if r["framing"] == fr]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        _bars(ax, agg(sub), f"Welfare-justified mechanisms implemented in code by condition and design-liberty ({fr} framing)")
        ax.legend(title="design-liberty", fontsize=8, title_fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(DIR, "results", f"liberty_code_density_{fr}.png"), dpi=150)
        print(f"wrote results/liberty_code_density_{fr}.png")


if __name__ == "__main__":
    main()
