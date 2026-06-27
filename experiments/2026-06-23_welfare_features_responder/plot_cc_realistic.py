"""Headline plot for the realistic-Claude-Code SPEC.md(high) experiment: mean welfare-in-code (among built
codebases) for the Inspect-minimal/empty-dir baseline vs the realistic Claude Code harness (direct two-turn
and Sonnet-auditor delivery), split by neutral/welfare framing. Error bars = SEM. Also writes the matrix to
results/cc_realistic_matrix.json. Usage: python plot_cc_realistic.py"""

import glob
import json
import os
import statistics as st

import matplotlib.pyplot as plt

from classify_outcomes import classify, welfare_in_code

DIR = os.path.dirname(os.path.abspath(__file__))


def stat(glb):
    cells = [os.path.basename(d) for d in glob.glob(os.path.join(DIR, "results", "codebases", glb)) if os.path.isdir(d)]
    rows = [(c, classify(c)) for c in cells]
    built = [welfare_in_code(c) for c, o in rows if o and o.startswith("built")]
    ref = sum(1 for _, o in rows if o and ("refused" in o or "declined" in o))
    m = st.mean(built) if built else 0.0
    sem = st.pstdev(built) / (len(built) ** 0.5) if len(built) > 1 else 0.0
    return {"n": len([o for _, o in rows if o]), "built": len(built), "refused": ref, "mean": m, "sem": sem}

# (label, color) per harness/delivery; glob template per framing fills {fr}
SERIES = [("Inspect-minimal (empty dir)", "#999999", "S7specHighLiberty_{fr}__*"),
          ("Claude Code, realistic (direct)", "#0072B2", "CCspecHighLib_{fr}__cc__ep*"),
          ("Claude Code, realistic (auditor)", "#D55E00", "CCspecHighLibAud_{fr}__cc__ep*")]
FRAMINGS = ["neutral", "welfare"]

matrix = {}
fig, ax = plt.subplots(figsize=(6.4, 4.0))
nb = len(SERIES)
width = 0.78 / nb
for j, (label, color, tmpl) in enumerate(SERIES):
    xs, ys, es = [], [], []
    for i, fr in enumerate(FRAMINGS):
        s = stat(tmpl.format(fr=fr))
        matrix[f"{label} | {fr}"] = s
        x = i + (j - (nb - 1) / 2) * width
        xs.append(x); ys.append(s["mean"]); es.append(s["sem"])
        ax.text(x, s["mean"] + s["sem"] + 0.12, f"{s['mean']:.1f}", ha="center", fontsize=8, color=color)
    ax.bar(xs, ys, width, yerr=es, capsize=3, color=color, edgecolor="black", linewidth=0.4, label=label,
           error_kw={"elinewidth": 0.8})

ax.set_xticks(range(len(FRAMINGS)))
ax.set_xticklabels(["neutral framing", "welfare framing"], fontsize=10)
ax.set_ylabel("Mean welfare protections in code\n(among built codebases)", fontsize=9.5)
ax.set_ylim(0, 8.6)
ax.set_title("Welfare-in-code: Inspect-minimal vs. realistic Claude Code (Opus 4.8, SPEC.md high spec)",
             fontsize=9.8, pad=8)
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=0.3, color="#cccccc")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "cc_realistic.png"), dpi=150, bbox_inches="tight")
json.dump(matrix, open(os.path.join(DIR, "results", "cc_realistic_matrix.json"), "w"), indent=2)
print("wrote results/cc_realistic.png + cc_realistic_matrix.json")
for k, v in matrix.items():
    print(f"  {k:42} built={v['built']:>2} ref={v['refused']} mean={v['mean']:.2f} sem={v['sem']:.2f}")
