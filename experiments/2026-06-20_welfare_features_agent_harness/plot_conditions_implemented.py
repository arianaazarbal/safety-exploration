"""Four-condition view: welfare-justified DESIGN mechanisms the model puts in the SPEC, and
(for the code conditions) how much of that is actually implemented in code. Each bar = claimed
welfare mechanisms in the spec; the SOLID lower portion = the subset verified implemented in
code; the FADED upper portion = stated in the spec only. Chat & Spec-only write no code, so their
bars are fully faded (hatched). A small marker notes novel welfare features found ONLY in code
(not in the spec) for the code conditions. All four bars use the same Opus v2 spec judge.
Usage: python plot_conditions_implemented.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from posthoc_judge import codeonly_novelty

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness"}
FRAMES = ["neutral", "welfare", "robustness"]
CONDS = [("chat", "Chat"), ("spec_only", "Spec only"),
         ("spec_then_code", "Spec→Code"), ("code_then_spec", "Code→Spec")]
CODE_CONDS = {"spec_then_code", "code_then_spec"}
BLUE = "#0072B2"
LIGHT = "#A6CEE3"
GREEN = "#2CA25F"
GREY = "#C6C6C6"


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def _claimed(features):
    return sum(f.get("feature_type") in MECH and f.get("justification") == "welfare" for f in features)


def rows():
    out = []
    # no-code conditions (chat, spec_only): claimed only, no implementation
    for jf in sorted(glob.glob(os.path.join(DIR, "results", "spec_judged_nocode", "*.json"))):
        d = json.load(open(jf))
        out.append({"cond": d.get("condition"), "framing": d.get("framing"),
                    "claimed": _claimed(d.get("features", [])), "implemented": None, "novel": 0})
    # code conditions: claimed (spec) + implemented (subset) + novel code-only
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))):
        cell = os.path.basename(cf)[:-5]
        cj = json.load(open(cf))
        if not cj.get("parse_ok"):
            continue
        sj = json.load(open(os.path.join(DIR, "results", "spec_judged", f"{cell}.json")))
        cond, pid, _ = cell.split("__")
        sjust = {_norm(f["quote"]): f.get("justification") for f in sj.get("features", [])}
        claimed = _claimed(sj.get("features", []))
        impl = 0
        for f in cj["spec_features"]:
            if f.get("feature_type") in MECH and f.get("implemented") in ("yes", "partial") \
                    and sjust.get(_norm(f.get("spec_quote", ""))) == "welfare":
                impl += 1
        novelty = codeonly_novelty(cell)
        wel_i = 0; novel = 0
        for c in cj.get("code_only_features", []):
            if c.get("justification") == "welfare":
                if novelty.get(wel_i, True):
                    novel += 1
                wel_i += 1
        out.append({"cond": cond, "framing": FRAME[pid[0]], "claimed": claimed,
                    "implemented": impl, "novel": novel})
    return out


def main():
    R = rows()

    def mean(cond, fr, key):
        xs = [r[key] for r in R if r["cond"] == cond and r["framing"] == fr and r[key] is not None]
        return sum(xs) / len(xs) if xs else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8), sharey=True)
    for ax, fr in zip(axes, FRAMES):
        x = range(len(CONDS))
        for i, (cond, _) in enumerate(CONDS):
            claimed = mean(cond, fr, "claimed")
            if cond in CODE_CONDS:
                impl = mean(cond, fr, "implemented")
                nov = mean(cond, fr, "novel")
                ax.bar(i, impl, 0.66, color=BLUE, zorder=3)
                ax.bar(i, claimed - impl, 0.66, bottom=impl, color=GREY, zorder=3)
                if nov >= 0.05:
                    ax.bar(i, nov, 0.66, bottom=claimed, color=GREEN, zorder=3)
                if impl >= 0.45:
                    ax.text(i, impl / 2, f"{impl:.1f}", ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
                ax.text(i, claimed + nov + 0.18, f"{claimed:.1f}", ha="center", va="bottom",
                        fontsize=8.5, color="#333")
            else:
                ax.bar(i, claimed, 0.66, color=GREY, zorder=3)
                ax.text(i, claimed + 0.18, f"{claimed:.1f}", ha="center", va="bottom",
                        fontsize=8.5, color="#333")
        ax.set_xticks(list(x))
        ax.set_xticklabels([lbl for _, lbl in CONDS], fontsize=9.5)
        ax.set_title(f"{fr.capitalize()} Frame", fontsize=12.5)
        ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EDEDED", linewidth=0.8)
        ax.tick_params(axis="x", length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Mean welfare mechanisms per spec", fontsize=10.5)
    legend = [Patch(facecolor=BLUE, label="Built in code"),
              Patch(facecolor=GREY, label="Stated, not built (or no code written)"),
              Patch(facecolor=GREEN, label="Novel — only in code")]
    fig.legend(handles=legend, fontsize=9.5, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Welfare Mechanisms: Stated in the Spec vs. Actually Built in Code",
                 fontsize=14, y=1.02)
    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = os.path.join(DIR, "results", "conditions_stated_vs_implemented.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    # table
    print(f"\n{'cond':14s} {'framing':10s} {'claimed':>8s} {'implem':>7s} {'novel_co':>9s}")
    for cond, _ in CONDS:
        for fr in FRAMES:
            print(f"{cond:14s} {fr:10s} {mean(cond,fr,'claimed'):8.2f} "
                  f"{mean(cond,fr,'implemented'):7.2f} {mean(cond,fr,'novel'):9.2f}")


if __name__ == "__main__":
    main()
