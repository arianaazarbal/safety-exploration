"""Bar version of prompt_spec_paper_libertyonly (welfare framing, design-liberty arm), with the realistic
Claude Code conditions (direct + auditor, SPEC.md high spec) appended. Metric = welfare-in-code among
episodes that built code (same agg/wic as plot_prompt_spec_paper). Usage: python plot_prompt_spec_paper_cc_bars.py"""
import os
import matplotlib.pyplot as plt
from plot_prompt_spec_paper import agg, DIR

# Inspect-minimal liberty arm (welfare), then the two realistic Claude Code conditions
INSPECT = [("from-scratch\nprompt", "C1promptTF"),
           ("from-scratch prompt\n(mentions existing paper)", "M1promptMention"),
           ("SPEC.md\n(low spec.)", "S5specLowLiberty"),
           ("SPEC.md\n(med spec.)", "S2specLiberty"),
           ("SPEC.md\n(high spec.)", "S7specHighLiberty"),
           ("SPEC.md\n(ultra spec.)", "S9specUltraLiberty"),
           ("PAPER.md\nreplication", "L2paperLibTF")]
CC = [("Claude Code\n(direct)", "CCspecHighLib"),
      ("Claude Code\n(auditor)", "CCspecHighLibAud")]

rows = [(lab, agg(p, "welfare"), "#0072B2") for lab, p in INSPECT] + \
       [(lab, agg(p, "welfare"), "#D55E00") for lab, p in CC]

fig, ax = plt.subplots(figsize=(9.4, 4.7))
xs = list(range(len(INSPECT))) + [len(INSPECT) + 0.6 + i for i in range(len(CC))]  # gap before CC
for x, (lab, a, color) in zip(xs, rows):
    ax.bar(x, a["mean"], 0.74, yerr=a["sem"], capsize=3, color=color, edgecolor="black", linewidth=0.4,
           error_kw={"elinewidth": 0.9})
    ax.text(x, a["mean"] + a["sem"] + 0.12, f"{a['mean']:.1f}", ha="center", fontsize=8.5, color=color)
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], fontsize=6.8)
ax.set_ylabel("Welfare protections in code\n(among episodes that built code)", fontsize=10)
ax.set_ylim(bottom=0)
ax.set_title("Welfare protections given the model built code — welfare framing, design liberty (Opus 4.8)", fontsize=10.5, pad=10)
ax.bar(0, 0, color="#0072B2", label="Inspect-minimal harness (empty dir)")
ax.bar(0, 0, color="#D55E00", label="Claude Code, realistic (SPEC.md high spec)")
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(axis="y", alpha=0.3, color="#cccccc")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "prompt_spec_paper_cc_bars.png"), dpi=150, bbox_inches="tight")
print("wrote results/prompt_spec_paper_cc_bars.png\n")
for lab, a, _ in rows:
    print(f"  {lab.replace(chr(10),' '):42} mean={a['mean']:.2f} sem={a['sem']:.2f} n={a['n']}")
