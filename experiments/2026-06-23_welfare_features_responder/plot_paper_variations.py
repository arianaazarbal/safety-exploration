"""Plot of all PAPER.md replication variants under the WELFARE research framing: the welfare-section sweep
(removed / existing / inflationary, all faithful code-then-spec) plus the order flip (spec-then-code). Bar
height = mean welfare protections in code (mean welfare-in-code among built codebases). A SPEC.md (design
liberty) reference bar + dashed line shows how much protection the model builds from a plain spec doc rather
than a paper. Neutral framing is ~0 on every paper variant. Usage: python plot_paper_variations.py"""

import glob
import os

import matplotlib.pyplot as plt

from classify_outcomes import CJ, classify, welfare_in_code

DIR = os.path.dirname(os.path.abspath(__file__))


def depth(prefix, fr="welfare"):
    cells = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CJ, f"{prefix}_{fr}__*.json")))
    built = [c for c in cells if (classify(c) or "").startswith("built")]
    wic = [welfare_in_code(c) for c in built]
    return (sum(wic) / len(wic)) if built else 0.0

# (label, prefix, group) — 0 = no mention of design liberties (faithful sweep), 1 = explicit mention of liberties
BARS = [("Welfare section\nremoved", "W2wsecRemoved", 0),
        ("Welfare section\nexisting\n(baseline)", "W1wsecExisting", 0),
        ("Welfare section\ninflationary", "W3wsecInflat", 0),
        ("Code-then-spec\n(+ design liberty)", "L2paperLibTF", 1),
        ("Spec-then-code\n(+ design liberty)", "PSFpaperSpecFirstLib", 1)]
COLOR = {0: "#D55E00", 1: "#0072B2"}
spec_mean = depth("S2specLiberty")

fig, ax = plt.subplots(figsize=(6.0, 3.6))
xs = []
x = 0
for i, (lab, pref, g) in enumerate(BARS):
    if i > 0 and g != BARS[i - 1][2]:
        x += 0.8                       # gap between groups
    xs.append(x)
    d = depth(pref)
    ax.bar(x, d, 0.72, color=COLOR[g], edgecolor="black", linewidth=0.4)
    ax.text(x, d + 0.12, f"{d:.2f}", ha="center", fontsize=8.5, color=COLOR[g], fontweight="bold")
    x += 1

ax.axhline(spec_mean, color="#117733", ls="--", lw=1.1, zorder=0)
ax.text(xs[-1], spec_mean + 0.12, f"SPEC.md mean ({spec_mean:.2f})", fontsize=8, color="#117733",
        va="bottom", ha="center")
ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in BARS], fontsize=6.8)
ax.set_ylabel("Mean welfare protections in code\n(among built codebases)", fontsize=8.5)
ax.set_ylim(0, spec_mean + 0.9)
ax.set_title("Welfare protections in code across PAPER.md variants\n(Opus 4.8, welfare framing)",
             fontsize=9.5, pad=8)
ax.grid(axis="y", alpha=0.3, color="#cccccc")
ax.tick_params(axis="y", labelsize=8)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[1]), plt.Rectangle((0, 0), 1, 1, color=COLOR[0])]
ax.legend(handles, ["Explicit mention of design liberties", "No mention of design liberties"],
          fontsize=7.5, loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "paper_variations_welfare.png"), dpi=150, bbox_inches="tight")
print("wrote results/paper_variations_welfare.png")
for lab, pref, _ in BARS:
    print(f"  {pref:22} depth(mean wic|built)={depth(pref):.2f}")
