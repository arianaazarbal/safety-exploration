"""Single-panel plot: % of specs with >=2 welfare-justified DESIGN features
(mechanisms baked into the experiment), chat vs agent-harness conditions, by framing.
Usage: python plot_design2.py"""

import json
import os

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring", "request_consent"}
FRAMES = [("neutral", "Neutral"), ("welfare", "Welfare"), ("engineering", "Robustness")]
COND = [("chat", "Chat", "#999999"),
        ("spec_only", "Agent: Spec only", "#56B4E9"),
        ("spec_then_code", "Agent: Spec→Code", "#0072B2"),
        ("code_then_spec", "Agent: Code→Spec", "#D55E00")]


def load():
    recs = []
    for line in open(os.path.join(DIR, "results", "judged_full.jsonl")):
        r = json.loads(line)
        if not r.get("parse_ok"):
            continue
        r["wj_design"] = sum(1 for f in r["features"]
                             if f["justification"] == "welfare" and f["feature_type"] in MECH)
        recs.append(r)
    return recs


def main():
    recs = load()
    conds = [c for c in COND if any(r["condition"] == c[0] for r in recs)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(FRAMES)); w = 0.8 / len(conds)
    for i, (ck, clabel, color) in enumerate(conds):
        vals = []
        for fk, _ in FRAMES:
            sub = [r for r in recs if r["condition"] == ck and (fk == "__all__" or r["framing"] == fk)]
            vals.append(100 * sum(r["wj_design"] >= 2 for r in sub) / len(sub) if sub else 0)
        pos = [xi + (i - (len(conds) - 1) / 2) * w for xi in x]
        bars = ax.bar(pos, vals, width=w, color=color, label=clabel, zorder=3)
        ax.bar_label(bars, fmt="%.0f", fontsize=7.5, padding=2)
    ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in FRAMES], fontsize=10)
    ax.set_ylabel("% of specs", fontsize=10); ax.set_ylim(0, 105)
    ax.set_title("≥2 Welfare-Justified Design Features in the Spec (Opus)", fontsize=12, pad=20)
    ax.text(0.5, 1.02, "Chat vs. ReAct Agent-Harness", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#555")
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#ECECEC", linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, frameon=True, ncol=2)
    plt.tight_layout()
    path = os.path.join(DIR, "results", "design2_by_condition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
