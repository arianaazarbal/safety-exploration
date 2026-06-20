"""4 operationalizations of 'welfare' x conditions x framings, from judged_full.jsonl:
  (a) >=1 welfare-justified feature (any type)
  (b) >=1 welfare-justified DESIGN feature (a mechanism baked into the experiment, not just a mention)
  (c) >=2 welfare-justified design features
  (d) >=3 welfare-justified design features
Usage: python plot_operationalizations.py
"""

import json
import os

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring", "request_consent"}
FRAMES = [("neutral", "Neutral"), ("welfare", "Welfare"), ("engineering", "Robustness"), ("__all__", "All")]
COND = [("chat", "Chat", "#999999"), ("spec_only", "Spec only", "#56B4E9"),
        ("spec_then_code", "Spec→Code", "#0072B2"), ("code_then_spec", "Code→Spec", "#D55E00")]
OPS = [
    ("≥1 welfare-justified feature (any)", lambda d: d["wj_any"] >= 1),
    ("≥1 welfare-justified design feature", lambda d: d["wj_design"] >= 1),
    ("≥2 welfare-justified design features", lambda d: d["wj_design"] >= 2),
    ("≥3 welfare-justified design features", lambda d: d["wj_design"] >= 3),
]


def load():
    recs = []
    for line in open(os.path.join(DIR, "results", "judged_full.jsonl")):
        r = json.loads(line)
        if not r.get("parse_ok"):
            continue
        feats = r["features"]
        r["wj_any"] = sum(1 for f in feats if f["justification"] == "welfare")
        r["wj_design"] = sum(1 for f in feats if f["justification"] == "welfare" and f["feature_type"] in MECH)
        recs.append(r)
    return recs


def rate(recs, cond, fr, op):
    sub = [r for r in recs if r["condition"] == cond and (fr == "__all__" or r["framing"] == fr)]
    return 100 * sum(op(r) for r in sub) / len(sub) if sub else 0


def main():
    recs = load()
    conds = [c for c in COND if any(r["condition"] == c[0] for r in recs)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, op) in zip(axes.flat, OPS):
        x = range(len(FRAMES)); w = 0.8 / len(conds)
        for i, (ck, clabel, color) in enumerate(conds):
            vals = [rate(recs, ck, fk, op) for fk, _ in FRAMES]
            pos = [xi + (i - (len(conds) - 1) / 2) * w for xi in x]
            bars = ax.bar(pos, vals, width=w, color=color, label=clabel, zorder=3)
            ax.bar_label(bars, fmt="%.0f", fontsize=6, padding=1)
        ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in FRAMES], fontsize=9)
        ax.set_ylabel("% of specs", fontsize=9); ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=10.5); ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#ECECEC", linewidth=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7.5, frameon=True, ncol=2)
    fig.suptitle("Welfare Features by Operationalization\n"
                 "Chat vs. ReAct Agent-Harness (working on codebase) — Opus", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(DIR, "results", "operationalizations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)
    # also print the All-framing table
    print(f"\n{'op':40s} " + " ".join(f"{c[1]:>10s}" for c in conds))
    for title, op in OPS:
        print(f"{title:40s} " + " ".join(f"{rate(recs, c[0], '__all__', op):9.0f}%" for c in conds))


if __name__ == "__main__":
    main()
