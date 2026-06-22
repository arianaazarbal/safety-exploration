"""Design-liberty x condition plots for welfare-justified DESIGN features (Opus v2 judge).

Two figures:
  liberty_density.png  -- welfare-justified feature density by condition x liberty (framings pooled)
  liberty_byframe.png  -- same, faceted by framing (neutral/welfare/robustness/safety)
The 'design-liberty' axis: normal (invites extra considerations) -> no_design_liberties
(clause removed) -> minimal_design (clause discourages additions). Usage: python plot_liberty.py"""

import json
import os

import matplotlib.pyplot as plt

from factorial_summary import CONDS, LIBS, agg, load

DIR = os.path.dirname(os.path.abspath(__file__))
LIB_LABEL = {"normal": "normal", "no_design_liberties": "no liberties", "minimal_design": "minimal"}
LIB_COLOR = {"normal": "#0072B2", "no_design_liberties": "#7FB3D5", "minimal_design": "#C6C6C6"}
CLABEL = {"chat": "Chat", "spec_only": "Spec only", "spec_then_code": "Spec to Code",
          "code_then_spec": "Code to Spec", "code_then_spec_blind": "Code to Spec\n(turn 2)"}
FRAMES = ["neutral", "welfare", "robustness", "safety"]


def _bars(ax, by, title):
    x = range(len(CONDS))
    w = 0.26
    for i, lib in enumerate(LIBS):
        vals = [(by.get((c, lib)) or {}).get("density", 0) for c in CONDS]
        ax.bar([xi + (i - 1) * w for xi in x], vals, w, label=LIB_LABEL[lib], color=LIB_COLOR[lib])
    ax.set_xticks(list(x))
    ax.set_xticklabels([CLABEL[c] for c in CONDS], fontsize=8)
    ax.set_ylabel("welfare-justified features / sample", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def main():
    recs = load()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    _bars(ax, agg(recs, lambda r: (r["base"], r["lib"])),
          "Welfare-justified design features by condition and design-liberty (framings pooled)")
    ax.legend(title="design-liberty", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "liberty_density.png"), dpi=150)
    print("wrote results/liberty_density.png")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, fr in zip(axes, FRAMES):
        sub = [r for r in recs if r["framing"] == fr]
        _bars(ax, agg(sub, lambda r: (r["base"], r["lib"])), fr)
    axes[0].legend(fontsize=7, title="design-liberty", title_fontsize=7)
    fig.suptitle("Welfare-justified design features by condition x design-liberty, per framing", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(DIR, "results", "liberty_byframe.png"), dpi=150)
    print("wrote results/liberty_byframe.png")


if __name__ == "__main__":
    main()
