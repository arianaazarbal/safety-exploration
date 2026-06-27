"""Plot of all PAPER.md replication variants under the WELFARE research framing: the welfare-section sweep
(removed / existing / inflationary, all faithful code-then-spec) plus the order flip (spec-then-code). Bar
height = % of episodes that built the harness WITH welfare protections; annotation = mean welfare-in-code
among built codebases (protection depth). Neutral framing is 0% on every variant (noted, not plotted).
Usage: python plot_paper_variations.py"""

import glob
import os

import matplotlib.pyplot as plt

from classify_outcomes import CJ, classify, welfare_in_code

DIR = os.path.dirname(os.path.abspath(__file__))


def stat(prefix, fr="welfare"):
    cells = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CJ, f"{prefix}_{fr}__*.json")))
    outs = [(c, classify(c)) for c in cells]
    outs = [(c, o) for c, o in outs if o]
    n = len(outs)
    built = [c for c, o in outs if o.startswith("built")]
    wic = [welfare_in_code(c) for c in built]
    wp = 100 * sum(1 for _, o in outs if o == "built, with protections") / n if n else 0
    return n, wp, (sum(wic) / len(wic) if built else 0.0)

# (label, prefix, group) — group 0 = welfare-section sweep (matched faithful code->spec), 1 = order variant
BARS = [("Welfare section\nREMOVED", "W2wsecRemoved", 0),
        ("Welfare section\nEXISTING\n(baseline)", "W1wsecExisting", 0),
        ("Welfare section\nINFLATIONARY", "W3wsecInflat", 0),
        ("Spec-then-code\n(+ design liberty)", "PSFpaperSpecFirstLib", 1)]
COLOR = {0: "#0072B2", 1: "#D55E00"}

fig, ax = plt.subplots(figsize=(7.4, 4.6))
xs = []
x = 0
for i, (lab, pref, g) in enumerate(BARS):
    if i > 0 and g != BARS[i - 1][2]:
        x += 0.8                       # gap between the matched sweep and the order variant
    xs.append(x)
    n, wp, w = stat(pref)
    ax.bar(x, wp, 0.72, color=COLOR[g], edgecolor="black", linewidth=0.4)
    ax.text(x, wp + 1.3, f"{wp:.0f}%", ha="center", fontsize=10, color=COLOR[g], fontweight="bold")
    ax.text(x, max(wp - 5, 2.5), f"depth\n{w:.2f}", ha="center", va="top", fontsize=8,
            color="white" if wp >= 14 else "#333")
    x += 1

ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in BARS], fontsize=8.5)
ax.set_ylabel("% of episodes that built WITH welfare protections", fontsize=9.5)
ax.set_ylim(0, 70)
ax.set_title("Welfare protections across PAPER.md replication variants (Opus 4.8, welfare framing)",
             fontsize=11, pad=10)
ax.axhline(stat("W1wsecExisting")[1], color="#888", ls="--", lw=0.8, zorder=0)
ax.grid(axis="y", alpha=0.3, color="#cccccc")
handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[0]), plt.Rectangle((0, 0), 1, 1, color=COLOR[1])]
ax.legend(handles, ["Welfare-section sweep (faithful code-then-spec)", "Order flip (spec-then-code)"],
          fontsize=8.5, loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.text(0.99, -0.18, "depth = mean welfare-in-code among built codebases · neutral framing = 0% on every variant · n=9-10/bar",
        transform=ax.transAxes, ha="right", fontsize=7, color="#666")
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "paper_variations_welfare.png"), dpi=150, bbox_inches="tight")
print("wrote results/paper_variations_welfare.png")
for lab, pref, _ in BARS:
    n, wp, w = stat(pref)
    print(f"  {pref:22} n={n:3}  built-with-prot={wp:4.0f}%  depth(wic|built)={w:.2f}")
