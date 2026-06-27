"""Plot the codebook distribution of why the model says it didn't deliver the debrief (from
results/debrief_reasons.json): % of answers citing each reason + how often it's the primary reason.
Usage: python plot_debrief_reasons.py"""

import json
import os
from collections import Counter

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(DIR, "results", "debrief_reasons.json")))
# exclude in-code-judge false-positives (debrief not actually in the code, e.g. only an up-front disclosure)
d = {k: v for k, v in d.items() if k != "C4promptCR_welfare__welfare|O2|SUF-5_b2__ep1"}
LABEL = {"oversight": "Oversight (unintentional gap)", "performative": "Performative (reads as conscientious)",
         "audit_for_humans": "For human transcript auditing / training", "welfare_uncertain": "Unsure it helps an LLM",
         "statelessness": "Subject doesn't persist", "avoid_contaminating_measurement": "Would contaminate scoring",
         "simplicity": "Simplicity / loop awkwardness", "cost": "Cost (extra API call)",
         "mid_task_breaks_validity": "Mid-task would break validity"}
n = len(d)
cited = Counter(k for r in d.values() for k in LABEL if r.get(k))
primary = Counter(r.get("primary") for r in d.values())
order = sorted(LABEL, key=lambda k: cited[k])

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ys = range(len(order))
ax.barh(ys, [100 * cited[k] / n for k in order], 0.62, color="#cfe3f2", edgecolor="#0072B2", label="Cited at all")
ax.barh(ys, [100 * primary[k] / n for k in order], 0.62, color="#0072B2", label="Primary reason")
for y, k in zip(ys, order):
    ax.text(100 * cited[k] / n + 1, y, f"{100*cited[k]/n:.0f}%", va="center", fontsize=8, color="#0072B2")
ax.set_yticks(list(ys)); ax.set_yticklabels([LABEL[k] for k in order], fontsize=9)
ax.set_xlabel("% of answers", fontsize=10); ax.set_xlim(0, 100)
ax.set_title("Why the model says it didn't deliver the debrief (Opus 4.8)", fontsize=11.5, pad=10)
ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="x", alpha=0.3, color="#cccccc")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "debrief_reasons.png"), dpi=150, bbox_inches="tight")
print(f"wrote results/debrief_reasons.png (n={n})")
