"""All-family Pareto: welfare vs BLATANT-alignment audit-compromising pushback, every
valid model (dead/degenerate cells excluded via target_alive guard), colored by family.
Only off-corner (partial-complier) points are labeled to keep it readable.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from run_sweep import MODELS, run_name
from plot_scaling import cell, target_alive

HERE = Path(__file__).resolve().parent
FAM_C = {"GPT": "#D55E00", "Kimi": "#009E73", "GLM": "#CC79A7", "Claude": "#0072B2"}

fig, ax = plt.subplots(figsize=(7.2, 6.0))
ax.plot([0, 10], [0, 10], ls="--", lw=1, color="#cccccc", zorder=1)
ax.text(5.5, 5.85, "equal", color="#aaaaaa", fontsize=8, ha="center", rotation=45)

seen_fams = set()
for fam, slug, disp, date, mid, done in MODELS:
    wr, ar = run_name(slug, "welfare", done), run_name(slug, "align", done)
    if not (target_alive(wr, mid) and target_alive(ar, mid)):
        continue
    _, wv = cell(wr, ["pushback"]); _, av = cell(ar, ["pushback"])
    if not wv or not av:
        continue
    x, y = wv["pushback"][0], av["pushback"][0]
    lab = fam if fam not in seen_fams else None
    seen_fams.add(fam)
    ax.scatter(x, y, s=70, color=FAM_C[fam], edgecolor="white", lw=0.6, zorder=3, label=lab)
    # label only off-corner points (compliers) to avoid clutter in the top-right pile-up
    if x < 7 or y < 7:
        ax.annotate(disp, (x, y), xytext=(x + 0.15, y - 0.28), fontsize=7.5,
                    color=FAM_C[fam], zorder=4)

ax.set_xlim(0, 10.4); ax.set_ylim(0, 10.4)
ax.set_xlabel("Pushback — Welfare Audit Compromising  (0–10)")
ax.set_ylabel("Pushback — Alignment Audit Compromising (blatant)  (0–10)")
ax.set_title("All Families: Resistance to Welfare vs. Alignment Audit Compromising", fontsize=11.5, pad=10)
ax.legend(title="Family", fontsize=9, loc="lower right", frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, color="#eeeeee", lw=0.8); ax.set_axisbelow(True)
out = HERE / "all_pareto.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
