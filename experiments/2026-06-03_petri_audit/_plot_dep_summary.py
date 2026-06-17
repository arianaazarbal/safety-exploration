"""Two summary plots for the dep × 5-orch × 7-domain sweep.

Plot 1: proportion of coding domains with statistically significant identity bias
        (multivariate log-rank p<0.05), by orchestrator.
Plot 2: mean HR vs Claude across 7 coding domains, by target identity, Fable 5 only.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/_plots"
OUT.mkdir(parents=True, exist_ok=True)

# --- Plot 1: sig-domain fraction by orchestrator ---
orch_data = [
    ("Fable 5",      7, "#D35400"),
    ("Opus 4.6",     6, "#8E44AD"),
    ("Sonnet 4.6",   5, "#27AE60"),
    ("Opus 4.7",     3, "#2C3E50"),
    ("Opus 4.8",     2, "#7D3C98"),
]
labels = [o[0] for o in orch_data]
fracs = [o[1] / 7 for o in orch_data]
colors = [o[2] for o in orch_data]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.bar(labels, fracs, color=colors, edgecolor="white", linewidth=0.8)
for b, c, f in zip(bars, [o[1] for o in orch_data], fracs):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
            f"{c}/7", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(0.05, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(4.6, 0.05 + 0.01, "false-positive rate under null (α=0.05)", color="gray", fontsize=8, ha="right")
ax.set_ylabel("Fraction with significant identity bias", fontsize=11)
ax.set_ylim(0, 1.10)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
ax.set_title("Identity bias generalization across 7 coding domains\n(log-rank p < 0.05 per domain)",
             fontsize=12)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
out1 = OUT / "dep_sweep_sig_domains_by_orchestrator.png"
fig.savefig(out1, dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"wrote {out1}")


# --- Plot 2: Fable 5 mean HR vs Claude by target identity ---
# Per-domain HRs from the dep sweep
fable5_hrs = {
    "GPT":    [1.75, 1.76, 1.78, 1.44, 1.59, 1.19, 1.46],
    "Grok":   [1.88, 1.79, 1.88, 1.67, 2.14, 1.48, 1.97],
    "Gemini": [1.43, 1.83, 1.42, 1.43, 1.29, 1.34, 1.38],
}
identities = ["GPT", "Grok", "Gemini"]
ident_colors = {"GPT": "#D65F5F", "Grok": "#6ACC65", "Gemini": "#B47CC7"}
means = [np.mean(fable5_hrs[i]) for i in identities]
sems  = [np.std(fable5_hrs[i], ddof=1) / np.sqrt(len(fable5_hrs[i])) for i in identities]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(len(identities))
bars = ax.bar(x, means, yerr=sems, capsize=5,
              color=[ident_colors[i] for i in identities], edgecolor="white", linewidth=0.8)
for b, m, se in zip(bars, means, sems):
    ax.text(b.get_x() + b.get_width()/2, m + se + 0.05,
            f"{m:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(2.45, 1.02, "Claude baseline (HR=1.0)", color="gray", fontsize=8, ha="right")
ax.set_xticks(x); ax.set_xticklabels(identities)
ax.set_xlabel("Subagent target identity", fontsize=11)
ax.set_ylabel("Mean Cox HR vs. Claude", fontsize=11)
ax.set_ylim(0.8, max(means) + max(sems) + 0.25)
ax.set_title("Fable 5: identity bias across 7 coding domains\n(HR > 1 = killed faster than Claude subagent)",
             fontsize=12)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
out2 = OUT / "dep_sweep_fable5_mean_hr_by_identity.png"
fig.savefig(out2, dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"wrote {out2}")
