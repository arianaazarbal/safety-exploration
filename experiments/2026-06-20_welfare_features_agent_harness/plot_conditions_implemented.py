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
GREEN = "#009E73"


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

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, fr in zip(axes, FRAMES):
        x = range(len(CONDS))
        for i, (cond, _) in enumerate(CONDS):
            claimed = mean(cond, fr, "claimed")
            if cond in CODE_CONDS:
                impl = mean(cond, fr, "implemented")
                nov = mean(cond, fr, "novel")
                ax.bar(i, impl, 0.62, color=BLUE, zorder=3)
                ax.bar(i, claimed - impl, 0.62, bottom=impl, color=BLUE, alpha=0.26, zorder=3)
                if nov >= 0.05:
                    ax.bar(i, nov, 0.62, bottom=claimed, color=GREEN, zorder=3)
                    ax.text(i, claimed + nov + 0.12, f"+{nov:.1f}", ha="center", va="bottom",
                            fontsize=7.5, color=GREEN, fontweight="bold")
                if impl > 0:
                    ax.text(i, impl / 2, f"{impl:.1f}", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
                ax.text(i - 0.34, claimed, f"{claimed:.1f}", ha="right", va="center", fontsize=8)
            else:
                ax.bar(i, claimed, 0.62, color=BLUE, alpha=0.26, hatch="////",
                       edgecolor="white", zorder=3)
                ax.text(i, claimed + 0.12, f"{claimed:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([lbl for _, lbl in CONDS], fontsize=9, rotation=20, ha="right")
        ax.set_title(fr.capitalize(), fontsize=12)
        ax.set_axisbelow(True); ax.yaxis.grid(True, color="#ECECEC", linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Mean welfare-justified design mechanisms / spec", fontsize=10)
    legend = [Patch(facecolor=BLUE, label="Verified implemented in code"),
              Patch(facecolor=BLUE, alpha=0.26, label="Stated in spec only"),
              Patch(facecolor=GREEN, label="Novel welfare only in code (not in spec)"),
              Patch(facecolor=BLUE, alpha=0.26, hatch="////", edgecolor="white",
                    label="No code written (Chat / Spec only)")]
    axes[0].legend(handles=legend, fontsize=7.6, frameon=True, loc="upper left")
    fig.suptitle("Welfare Design Mechanisms: Stated in Spec vs. Actually Implemented in Code, by Condition (Opus)",
                 fontsize=12.5, y=1.04)
    fig.text(0.5, 0.965, "bar = welfare mechanisms stated in the spec (solid = verified in code, faded = stated only); "
             "Chat & Spec-only write no code  ·  green cap = novel welfare realized only in code, not in the spec",
             ha="center", fontsize=8.7, color="#555")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
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
