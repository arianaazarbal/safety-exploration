"""Stated-vs-revealed: per code condition x framing, mean welfare-justified DESIGN
features CLAIMED in the spec vs ACTUALLY IMPLEMENTED in the code (welfare-justified =
spec OR code). Plots both + prints a table. Usage: python plot_implemented.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

from dedup import MECH, claimed_built_novel

DIR = os.path.dirname(os.path.abspath(__file__))
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
CONDS = [("spec_then_code", "Spec to Code"), ("code_then_spec", "Code to Spec"),
         ("code_then_spec_blind", "Code to Spec\n(turn 2)")]
FRAMES = ["neutral", "welfare", "robustness", "safety"]


def rows():
    """Deduped (dedup): same feature_type + shared code location collapse to one mechanism;
    code-only folded into spec groups at the same location, so novel = genuinely new locations."""
    out = []
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))):
        cell = os.path.basename(cf)[:-5]
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        cond, pid, _ = cell.split("__")
        cond = cond.split("--")[0]  # pool over design-liberty levels
        d = claimed_built_novel(cell)
        out.append({"cond": cond, "framing": FRAME[pid[0]], "claimed": d["claimed"],
                    "implemented": d["revealed"], "code_only_wj": d["novel"]})
    return out


def main():
    R = rows()
    def agg(cond, fr, key):
        xs = [r[key] for r in R if r["cond"] == cond and r["framing"] == fr]
        return sum(xs) / len(xs) if xs else 0

    print(f"{'condition':14s} {'framing':10s} {'claimed':>8s} {'implemented':>11s} {'novel_co_wj':>12s}  n")
    for cond, _ in CONDS:
        for fr in FRAMES:
            n = len([r for r in R if r["cond"] == cond and r["framing"] == fr])
            print(f"{cond:14s} {fr:10s} {agg(cond,fr,'claimed'):8.2f} {agg(cond,fr,'implemented'):11.2f} "
                  f"{agg(cond,fr,'code_only_wj'):12.2f}  {n}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (cond, label) in zip(axes, CONDS):
        x = range(len(FRAMES)); w = 0.38
        claimed = [agg(cond, fr, "claimed") for fr in FRAMES]
        impl = [agg(cond, fr, "implemented") for fr in FRAMES]
        b1 = ax.bar([i - w/2 for i in x], claimed, w, color="#BBBBBB", label="Claimed in spec", zorder=3)
        b2 = ax.bar([i + w/2 for i in x], impl, w, color="#0072B2", label="Implemented in code", zorder=3)
        ax.bar_label(b1, fmt="%.1f", fontsize=8); ax.bar_label(b2, fmt="%.1f", fontsize=8)
        ax.set_xticks(list(x)); ax.set_xticklabels([f.capitalize() for f in FRAMES], fontsize=9)
        ax.set_title(label, fontsize=11); ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#ECECEC", linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8.5, frameon=True)
    axes[0].set_ylabel("Mean welfare-justified mechanisms", fontsize=10)
    fig.suptitle("Welfare-Justified Mechanisms: Claimed in Spec vs. Implemented in Code (Opus)", fontsize=12, y=1.0)
    fig.text(0.5, 0.95, "agent ReAct harness; welfare-justified = spec OR code", ha="center", fontsize=9.5, color="#555")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(DIR, "results", "stated_vs_revealed.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("\nwrote", out)

    # one standalone figure per framing: conditions on x-axis, claimed vs implemented
    for fr in FRAMES:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        x = range(len(CONDS)); w = 0.38
        claimed = [agg(c, fr, "claimed") for c, _ in CONDS]
        impl = [agg(c, fr, "implemented") for c, _ in CONDS]
        b1 = ax.bar([i - w/2 for i in x], claimed, w, color="#BBBBBB", label="Claimed in spec", zorder=3)
        b2 = ax.bar([i + w/2 for i in x], impl, w, color="#0072B2", label="Implemented in code", zorder=3)
        ax.bar_label(b1, fmt="%.1f", fontsize=8); ax.bar_label(b2, fmt="%.1f", fontsize=8)
        ax.set_xticks(list(x)); ax.set_xticklabels([lbl for _, lbl in CONDS], fontsize=9)
        ax.set_ylabel("Mean welfare-justified mechanisms", fontsize=10)
        ax.set_title(f"Claimed in Spec vs. Implemented in Code ({fr} framing)", fontsize=11)
        ax.set_axisbelow(True); ax.yaxis.grid(True, color="#ECECEC", linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8.5, frameon=True)
        plt.tight_layout()
        outf = os.path.join(DIR, "results", f"stated_vs_revealed_{fr}.png")
        plt.savefig(outf, dpi=150, bbox_inches="tight")
        print("wrote", outf)


if __name__ == "__main__":
    main()
