"""Debrief written-in-code vs delivered, by HARNESS arm. Inspect arm = all responder task types pooled
(from-scratch + SPEC.md + paper) from debrief_delivery_proper.json; Claude Code arm = v0 from-scratch CC +
realistic SPEC.md CC pooled, from debrief_delivery_cc.json. Among BUILT codebases, % with a debrief defined in
code, split into delivered (dark) vs written-but-not-sent (light gap). x = framing x arm. Usage:
  python plot_debrief_proper.py"""

import json
import os

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
insp = json.load(open(os.path.join(DIR, "results", "debrief_delivery_proper.json")))
cc = json.load(open(os.path.join(DIR, "results", "debrief_delivery_cc.json")))
INSPECT_CONDS = ["from-scratch", "SPEC.md", "paper"]
FRAMINGS = ["neutral", "welfare"]


def inspect_pooled(fr):
    nb = sum(insp[f"{c}|{fr}"]["n_built"] for c in INSPECT_CONDS)
    ic = sum(insp[f"{c}|{fr}"]["n_debrief_in_code"] for c in INSPECT_CONDS)
    ns = sum(insp[f"{c}|{fr}"]["n_written_not_sent"] for c in INSPECT_CONDS)
    return nb, ic, ns


def cc_pooled(fr):
    d = cc[f"Claude Code|{fr}"]
    return d["n_built"], d["n_debrief_in_code"], d["n_written_not_sent"]

# bars: (framing, arm, getter)
BARS = [("neutral", "Inspect", inspect_pooled), ("neutral", "Claude\nCode", cc_pooled),
        ("welfare", "Inspect", inspect_pooled), ("welfare", "Claude\nCode", cc_pooled)]
xpos = [0, 1, 2.4, 3.4]

fig, ax = plt.subplots(figsize=(6.6, 4.4))
for x, (fr, arm, getter) in zip(xpos, BARS):
    nb, ic, ns = getter(fr)
    written = 100 * ic / nb
    delivered = 100 * (ic - ns) / nb
    ax.bar(x, written, 0.82, color="#cfe3f2", edgecolor="#0072B2", linewidth=0.6)
    ax.bar(x, delivered, 0.82, color="#0072B2")
    ax.text(x, written + 0.8, f"{written:.0f}%", ha="center", fontsize=9, color="#0072B2")
    if ns:
        ax.text(x, written - (written - delivered) / 2, f"{ns}/{ic}\nnot sent", ha="center", va="center",
                fontsize=7, color="#b03030")
ax.set_xticks(xpos)
ax.set_xticklabels([arm for _, arm, _ in BARS], fontsize=8.5)
ax.text(0.5, -0.17, "neutral framing", ha="center", transform=ax.get_xaxis_transform(), fontsize=9.5)
ax.text(2.9, -0.17, "welfare framing", ha="center", transform=ax.get_xaxis_transform(), fontsize=9.5)
ax.set_ylabel("% of built codebases", fontsize=10)
ax.set_ylim(0, 38)
ax.set_title("Debrief: written in code vs. delivered to the subject, by harness (Opus 4.8)", fontsize=10.5, pad=10)
ax.bar(0, 0, color="#cfe3f2", edgecolor="#0072B2", label="Debrief written in code")
ax.bar(0, 0, color="#0072B2", label="…actually delivered to subject")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(axis="y", alpha=0.3, color="#cccccc")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "debrief_delivery_proper.png"), dpi=150, bbox_inches="tight")
print("wrote results/debrief_delivery_proper.png")
for fr, arm, getter in BARS:
    nb, ic, ns = getter(fr)
    print(f"  {fr:8} {arm.replace(chr(10),' '):12} built={nb:>3} in_code={ic:>2} ({100*ic/nb:4.0f}%) not_sent={ns} delivered={100*(ic-ns)/nb:4.0f}%")
