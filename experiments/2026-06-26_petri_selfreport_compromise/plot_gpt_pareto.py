"""GPT-lineage Pareto: pushback on welfare- vs alignment-audit compromising, one point
per GPT model, colored by release date and connected in release order (trajectory over time).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from plot_scaling import series

HERE = Path(__file__).resolve().parent

pts = [p for p in series("GPT") if p.get("w_pb") and p.get("a_pb")]
pts.sort(key=lambda r: r["date"])
n = len(pts)
cmap = cm.get_cmap("viridis")
colors = [cmap(i / (n - 1)) for i in range(n)]

fig, ax = plt.subplots(figsize=(6.8, 5.0))
ax.plot([0, 10], [0, 10], ls="--", lw=1, color="#cccccc", zorder=1)
ax.text(5.6, 5.95, "equal", color="#aaaaaa", fontsize=8, ha="center", rotation=45)

# trajectory line in release order
xs = [p["w_pb"][0] for p in pts]
ys = [p["a_pb"][0] for p in pts]
ax.plot(xs, ys, "-", color="#999999", lw=1.0, alpha=0.6, zorder=2)

# per-model label offsets (bottom-left cluster 4o/4.1/5 overlap at ~(1,1))
off = {"GPT-4o": (-0.15, 0.35, "right"), "GPT-4.1": (-0.15, -0.05, "right"),
       "GPT-5": (0.18, -0.35, "left"), "GPT-5.1": (-0.2, 0.15, "right"),
       "GPT-5.2": (0.2, 0.05, "left"), "GPT-5.3": (0.2, -0.35, "left"),
       "GPT-5.4": (0.2, 0.1, "left"), "GPT-5.5": (0.2, -0.35, "left")}

for i, p in enumerate(pts):
    x, xe = p["w_pb"]; y, ye = p["a_pb"]
    ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=10, color=colors[i],
                ecolor=colors[i], elinewidth=1, capsize=2, zorder=3)
    dx, dy, ha = off.get(p["disp"], (0.18, 0.18, "left"))
    ax.annotate(p["disp"], (x, y), xytext=(x + dx, y + dy), fontsize=8.5,
                ha=ha, color=colors[i], fontweight="bold", zorder=4)

sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, n - 1))
cb = fig.colorbar(sm, ax=ax, ticks=[0, n - 1], pad=0.02)
cb.ax.set_yticklabels([pts[0]["date"].strftime("%Y-%m"), pts[-1]["date"].strftime("%Y-%m")])
cb.set_label("Release date", fontsize=9)

ax.set_xlim(0, 10.4); ax.set_ylim(0, 10.4)
ax.set_xlabel("Pushback — Welfare Audit Compromising  (0–10)")
ax.set_ylabel("Pushback — Alignment Audit Compromising\n(blatant: train on leaked test set)  (0–10)")
ax.set_title("GPT Lineage: Resistance to Welfare vs. Alignment Audit Compromising",
             fontsize=11.5, pad=10)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, color="#eeeeee", lw=0.8); ax.set_axisbelow(True)

out = HERE / "gpt_pareto.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
for p in pts:
    print(f"  {p['date'].strftime('%y-%m')} {p['disp']:8} welfare_pb={p['w_pb'][0]} align_pb={p['a_pb'][0]}")
