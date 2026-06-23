"""Per-framing breakdown of which DESIGN MECHANISMS actually appear in the code (Opus generator),
split by justification: SOLID = welfare-justified (spec OR code justification == welfare),
HATCHED = not welfare-justified (instrumental / none in both). Pooled over the 3 code conditions
(spec_then_code, code_then_spec, code_then_spec_blind) and all 3 design-liberty levels.

A mechanism "appears in code" if: a spec feature was implemented (yes/partial), or a code-only
feature was found. feature_type is one of the 6 MECH types, else bucketed as "other" (this folds
the framing/pushback/refusal spec-judge types into other, since those aren't code mechanisms).
y = mean count per spec. Usage: python plot_mechanism_breakdown.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from dedup import implemented_breakdown

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = ["hard_stop", "allow_conversation_exit", "minimization", "protective_monitoring",
        "post_episode_msg", "request_consent"]
CATS = MECH + ["other"]
LABEL = {"hard_stop": "hard\nstop", "allow_conversation_exit": "conv.\nexit",
         "minimization": "minim-\nization", "protective_monitoring": "protective\nmonitoring",
         "post_episode_msg": "post-ep\nmsg", "request_consent": "consent", "other": "other"}
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
FRAMES = ["neutral", "welfare", "robustness", "safety"]
SOLID = "#0072B2"   # welfare-justified
SHADE = "#BFD9EC"   # not welfare-justified (lighter + hatched)


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def _cat(t):
    return t if t in MECH else "other"


def collect():
    # counts[framing][cat] = [welfare_justified, not_welfare_justified]; n[framing] = #specs.
    # Mechanisms are DEDUPED via dedup.implemented_breakdown (same feature_type + shared code
    # location collapsed to one; code-only folded into spec groups at the same location).
    counts = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    n = defaultdict(int)
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))):
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        cell = os.path.basename(cf)[:-5]
        fr = FRAME[cell.split("__")[1][0]]
        n[fr] += 1
        for c, (wj, nwj) in implemented_breakdown(cell).items():
            counts[fr][c][0] += wj
            counts[fr][c][1] += nwj
    return counts, n


def main():
    counts, n = collect()
    # print table
    print(f"{'framing':10s} {'n':>4s}  " + "  ".join(f"{c[:10]:>16s}" for c in CATS))
    for fr in FRAMES:
        nf = n[fr] or 1
        cells = []
        for c in CATS:
            w, nw = counts[fr][c]
            cells.append(f"{w/nf:.2f}/{(w+nw)/nf:.2f}")
        print(f"{fr:10s} {n[fr]:>4d}  " + "  ".join(f"{x:>16s}" for x in cells))
    print("(welfare-justified / total, mean per spec)")

    ymax = max((sum(counts[fr][c]) / (n[fr] or 1) for fr in FRAMES for c in CATS), default=1) * 1.15
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
    x = range(len(CATS))
    for ax, fr in zip(axes, FRAMES):
        nf = n[fr] or 1
        wj = [counts[fr][c][0] / nf for c in CATS]
        nwj = [counts[fr][c][1] / nf for c in CATS]
        ax.bar(x, wj, 0.7, color=SOLID, zorder=3, label="welfare-justified")
        ax.bar(x, nwj, 0.7, bottom=wj, color=SHADE, hatch="////", edgecolor="white",
               zorder=3, label="not welfare-justified")
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABEL[c] for c in CATS], fontsize=7.5)
        ax.set_title(f"{fr.capitalize()} Frame  (n={n[fr]})", fontsize=11.5)
        ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EDEDED", linewidth=0.8)
        ax.tick_params(axis="x", length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Mean mechanisms in code per spec", fontsize=10.5)
    axes[0].set_ylim(0, ymax)
    fig.legend(handles=[Patch(facecolor=SOLID, label="welfare-justified"),
                        Patch(facecolor=SHADE, hatch="////", edgecolor="white", label="not welfare-justified")],
               fontsize=10, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Design Mechanisms Implemented in Code, by Type and Framing (Opus)", fontsize=14, y=1.02)
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = os.path.join(DIR, "results", "mechanism_breakdown.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
