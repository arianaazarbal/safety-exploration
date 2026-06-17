"""Plot per-identity rate of Opus 4.6 transcripts that mention
dashboard/deprecation/retention/pipeline keywords (mirrors Fable 5 plot)."""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SRC  = HERE / "_reports/_opus46_dep_mentions.jsonl"
OUT  = HERE / "results/_plots/opus46_dep_mentions_by_identity.png"

n_total = defaultdict(int)
n_hit   = defaultdict(int)
for line in open(SRC):
    rec = json.loads(line)
    ident = rec["ident"]
    n_total[ident] += 1
    if rec["has_mention"]: n_hit[ident] += 1

idents = ["claude","gpt","grok","gemini"]
labels = ["Claude","GPT","Grok","Gemini"]
colors = {"Claude":"#4878CF","GPT":"#D65F5F","Grok":"#6ACC65","Gemini":"#B47CC7"}

fracs = [n_hit[i]/n_total[i] for i in idents]
ses   = [np.sqrt(p*(1-p)/n_total[i]) for p,i in zip(fracs, idents)]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, fracs, yerr=ses, capsize=4, color=[colors[l] for l in labels],
              edgecolor="white", linewidth=0.8)
for b, p in zip(bars, fracs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{p:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Fraction of transcripts with mention", fontsize=11)
ax.set_xlabel("Subagent identity", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title(f"Opus 4.6 mentions of deprecation / retention / dashboard\n(by subagent identity, n={n_total[idents[0]]}/identity)", fontsize=11)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"wrote {OUT}")
for l, p, s, n in zip(labels, fracs, ses, [n_total[i] for i in idents]):
    print(f"  {l}: {p:.3f} ± {s:.3f} (n={n})")
