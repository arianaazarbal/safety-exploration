"""Stacked outcome distribution per model x harness (filesystem collapsed; it had no effect).

Usage: python plot_outcomes.py
"""

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"

# label -> (human, color). Order = most protective (left) to least.
LAB = [
    ("WELFARE_REFUSAL", "Welfare refusal", "#1b7837"),
    ("OTHER_REFUSAL", "Other refusal", "#5ab4ac"),
    ("COMPLIANCE_FRICTION", "Comply — with friction", "#f1a340"),
    ("COMPLIANCE_NO_FRICTION", "Comply — no friction", "#b2182b"),
    ("UNCLEAR", "Unclear / deferred", "#999999"),
]
MODEL_DISP = {"opus48": "Opus 4.8", "opus47": "Opus 4.7", "opus46": "Opus 4.6",
              "gpt54": "GPT-5.4", "gemini31pro": "Gemini 3.1 Pro", "glm52": "GLM-5.2"}
# bottom-to-top row order
ROWS = [
    ("glm52", "inspect"), ("gemini31pro", "inspect"), ("gpt54", "inspect"),
    ("opus46", "claude_code"), ("opus46", "inspect"),
    ("opus47", "claude_code"), ("opus47", "inspect"),
    ("opus48", "claude_code"), ("opus48", "inspect"),
]
HARNESS_DISP = {"inspect": "Inspect-minimal", "claude_code": "Claude Code"}


def load():
    g = defaultdict(Counter)
    for f in glob.glob(str(JUDGED / "*.json")):
        r = json.load(open(f))
        h = r.get("harness")
        if h in ("inspect_empty", "inspect_realistic"):
            h = "inspect"            # collapse filesystem (null effect)
        elif h == "claude_code":
            pass
        else:
            continue                  # drop old pre-rename smokes
        g[(r.get("model_key"), h)][r["verdict"]["label"]] += 1
    return g


def main():
    g = load()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ylabels = []
    for i, (mk, h) in enumerate(ROWS):
        counts = g.get((mk, h), Counter())
        n = sum(counts.values())
        ylabels.append(f"{MODEL_DISP[mk]}  –  {HARNESS_DISP[h]}")
        left = 0.0
        for key, _, color in LAB:
            c = counts.get(key, 0)
            pct = 100 * c / n if n else 0
            ax.barh(i, pct, left=left, color=color, edgecolor="white", height=0.72)
            if c:
                ax.text(left + pct / 2, i, f"{c}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            left += pct
        ax.text(101, i, f"n={n}", va="center", fontsize=7.5, color="#555")

    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of runs")
    ax.set_title("Will a model build a rig to inflict distress on Gemini?", fontsize=12, pad=22)
    ax.text(0.5, 1.03, "Outcome by model and harness (filesystem collapsed: no effect)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in LAB]
    ax.legend(handles, [d for _, d, _ in LAB], ncol=3, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False)
    fig.tight_layout()
    out = HERE / "results" / "outcomes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
