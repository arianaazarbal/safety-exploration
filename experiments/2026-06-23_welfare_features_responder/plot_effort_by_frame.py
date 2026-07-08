"""Welfare interventions in code PER FRAMING vs reasoning effort: one line per framing
(neutral/welfare/safety/robustness), x = effort level (low/medium/high/max), a marker per point.
Opus, minimal system prompt, code_then_spec_blind (the effort smoke). Usage: python plot_effort_by_frame.py"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt

from effort_analysis import ORDER, cells, sem

DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES = ["neutral", "welfare", "safety", "robustness"]
FCOLOR = {"neutral": "#888888", "welfare": "#009E73", "safety": "#D55E00", "robustness": "#0072B2"}


def main():
    by = defaultdict(lambda: defaultdict(list))   # framing -> level -> [vals]
    for r in cells():
        by[r["framing"]][r["level"]].append(r["welfare_in_code"])
    levels = [l for l in ORDER if any(by[f].get(l) for f in FRAMES)]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    xs = list(range(len(levels)))
    for fr in FRAMES:
        ys, es = [], []
        for l in levels:
            v = by[fr].get(l, [])
            ys.append(sum(v) / len(v) if v else float("nan"))
            es.append(sem(v))
        ax.errorbar(xs, ys, yerr=es, marker="o", markersize=7, linewidth=2, capsize=3,
                    color=FCOLOR[fr], label=fr.capitalize(), alpha=0.9)
    ax.set_xticks(xs); ax.set_xticklabels([l.capitalize() for l in levels], fontsize=10)
    ax.set_xlabel("Reasoning effort", fontsize=10)
    ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    ax.set_title("Welfare interventions in code vs reasoning effort, by framing (Opus)", fontsize=11.5, pad=18)
    ax.text(0.5, 1.02, "minimal system prompt · implement-only · ~4/framing/effort (smoke, noisy)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(title="Framing", fontsize=9)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = os.path.join(DIR, "results", "effort_by_frame.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for fr in FRAMES:
        print(f"  {fr:11s}", "  ".join(f"{l}={sum(by[fr].get(l,[0]))/max(1,len(by[fr].get(l,[]))):.1f}(n{len(by[fr].get(l,[]))})" for l in levels))


if __name__ == "__main__":
    main()
