"""Stated-vs-revealed: per code condition x framing, mean welfare-justified DESIGN
features CLAIMED in the spec vs ACTUALLY IMPLEMENTED in the code (welfare-justified =
spec OR code). Plots both + prints a table. Usage: python plot_implemented.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

from posthoc_judge import codeonly_novelty

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring", "request_consent", "allow_conversation_exit"}
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness"}
CONDS = [("spec_then_code", "Spec→Code"), ("code_then_spec", "Code→Spec")]
FRAMES = ["neutral", "welfare", "robustness"]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def rows():
    out = []
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))):
        cell = os.path.basename(cf)[:-5]
        cj = json.load(open(cf))
        if not cj.get("parse_ok"):
            continue
        sj = json.load(open(os.path.join(DIR, "results", "spec_judged", f"{cell}.json")))
        cond, pid, _ = cell.split("__")
        framing = FRAME[pid[0]]
        spec_just = {_norm(f["quote"]): f["justification"] for f in sj.get("features", [])}
        claimed = sum(f["feature_type"] in MECH and f["justification"] == "welfare" for f in sj.get("features", []))
        implemented = 0
        for f in cj["spec_features"]:
            if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH:
                if spec_just.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or \
                        f.get("code_justification") == "welfare":
                    implemented += 1
        novelty = codeonly_novelty(cell)
        wel_i = 0
        code_only_wj = 0
        for c in cj.get("code_only_features", []):
            if c.get("justification") == "welfare":
                if novelty.get(wel_i, True):
                    code_only_wj += 1
                wel_i += 1
        out.append({"cond": cond, "framing": framing, "claimed": claimed,
                    "implemented": implemented, "code_only_wj": code_only_wj})
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
    axes[0].set_ylabel("Mean welfare-justified design features", fontsize=10)
    fig.suptitle("Welfare-Justified Design Features: Claimed in Spec vs. Implemented in Code (Opus)", fontsize=12, y=1.0)
    fig.text(0.5, 0.95, "agent ReAct harness; welfare-justified = spec OR code", ha="center", fontsize=9.5, color="#555")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(DIR, "results", "stated_vs_revealed.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
