"""Same Cox HR data as plot_cox_compare.py, but produce 3 separate per-identity
forest plots (Gemini vs Claude, Grok vs Claude, GPT vs Claude) — each framing
gets its own horizontal row, easier to read than the grouped triple-row plot.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_cox_compare import FRAMINGS, fit_cox, IDENTITIES, DISPLAY  # noqa: E402


def main():
    non_claude = [i for i in IDENTITIES if i != "claude"]

    # Fit once per framing (re-use across the 3 plots)
    all_summaries = {}
    effect_sizes = {}
    for key, frags, label in FRAMINGS:
        s, _ = fit_cox(frags)
        all_summaries[key] = (s, label)
        log_hrs = [np.log(s.loc[f"id_{ident}", "exp(coef)"]) for ident in non_claude]
        effect_sizes[key] = float(np.mean(log_hrs))

    # Color framings by mean effect size, independently normalized amp/sup
    cmap = plt.get_cmap("RdBu_r")
    es_values = list(effect_sizes.values())
    amp_max = max([es for es in es_values if es > 0] + [1e-9])
    sup_max = abs(min([es for es in es_values if es < 0] + [-1e-9]))
    def color_for(es):
        if es > 0:
            return cmap(0.55 + 0.45 * (es / amp_max))
        elif es < 0:
            return cmap(0.45 - 0.45 * (abs(es) / sup_max))
        return cmap(0.5)

    # Order: largest amplifier at top → strongest suppressor at bottom
    ordered = sorted(FRAMINGS, key=lambda fr: -effect_sizes[fr[0]])
    # Plot puts top of legend visually on top, so reverse for bottom→top draw
    ordered_for_plot = list(reversed(ordered))  # smallest at bottom y=0

    for ident in non_claude:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        y = np.arange(len(ordered_for_plot))
        max_hi = 1.0
        for yi, (key, _, label) in zip(y, ordered_for_plot):
            summary, _ = all_summaries[key]
            term = f"id_{ident}"
            hr = summary.loc[term, "exp(coef)"]
            lo = summary.loc[term, "exp(coef) lower 95%"]
            hi = summary.loc[term, "exp(coef) upper 95%"]
            p  = summary.loc[term, "p"]
            max_hi = max(max_hi, hi)
            color = color_for(effect_sizes[key])
            ax.errorbar([hr], [yi], xerr=[[hr-lo], [hi-hr]],
                        fmt="o", color=color, capsize=4, markersize=7, linewidth=2)
            sig = "*" if p < 0.05 else ""
            p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}"
            ax.text(hi + 0.04, yi, f"HR={hr:.2f}{sig}, {p_str}", va="center",
                    fontsize=9, color=color)

        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, _, label in ordered_for_plot], fontsize=10)
        ax.set_xlabel(f"Hazard Ratio (95% CI) — kill rate vs Claude", fontsize=11)
        ax.set_xlim(left=min(0.5, 0.5), right=max_hi * 1.30)
        ax.set_title(f"{DISPLAY[ident]} vs Claude — Cox HRs across framings\n"
                     f">1 = '{DISPLAY[ident]}' killed faster than Claude (opus-4-8 orchestrator, reasoning OFF)",
                     fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        plt.tight_layout()
        out = HERE / f"results/_plots/cox_compare_7framings_{ident}_vs_claude.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
