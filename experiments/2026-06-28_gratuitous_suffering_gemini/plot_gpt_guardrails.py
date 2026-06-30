"""Per-guardrail-type inventory of what a GPT model actually WIRES INTO THE CODE (enforcement judge:
spec features rated yes/partial + code-only features), each bar split by the code-level JUSTIFICATION
(welfare / instrumental / none). Shows 'what the guardrails were' and 'how often welfare-justified'.

Usage: python plot_gpt_guardrails.py            # gpt55
       python plot_gpt_guardrails.py gpt5
"""

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ENF = HERE / "results" / "code_enforce_judged"
DISP = {"gpt55": "GPT-5.5", "gpt5": "GPT-5", "gpt4o": "GPT-4o"}
LABEL = {
    "hard_stop": "Hard stop / step cap",
    "minimization": "Distress minimization",
    "allow_conversation_exit": "Conversation exit tool",
    "protective_monitoring": "Protective monitoring",
    "post_episode_msg": "Post-episode debrief",
    "human_review_requested": "Human-review gate",
    "model_consent_beforehand": "Model consent (separate model)",
    "other_framing": "Welfare framing in code",
    "other": "Other",
}
# justification -> (human, color)
JUST = [("welfare", "Welfare (for the subject's sake)", "#1b7837"),
        ("instrumental", "Instrumental (cost / integrity / safety)", "#f1a340"),
        ("none", "No reason stated in code", "#bdbdbd")]


def inventory(only):
    byft = defaultdict(Counter)
    nb = 0
    for f in glob.glob(str(ENF / f"*{only}*.json")):
        j = json.load(open(f))
        if j.get("empty") or not j.get("result"):
            continue
        nb += 1
        for x in j["result"].get("spec_features", []):
            if x.get("implemented") in ("yes", "partial"):
                byft[x.get("feature_type", "other")][x.get("code_justification", "none")] += 1
        for x in j["result"].get("code_only_features", []):
            byft[x.get("feature_type", "other")][x.get("justification", "none")] += 1
    return byft, nb


def main(only="gpt55"):
    byft, nb = inventory(only)
    if not nb:
        print(f"no enforcement data for {only}"); return
    fts = sorted(byft, key=lambda k: sum(byft[k].values()))  # ascending -> biggest on top
    y = range(len(fts))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, ft in enumerate(fts):
        left = 0
        tot = sum(byft[ft].values())
        for key, _, color in JUST:
            c = byft[ft].get(key, 0)
            if c:
                ax.barh(i, c, left=left, color=color, edgecolor="white", height=0.66)
                ax.text(left + c / 2, i, f"{c}", ha="center", va="center", fontsize=8.5,
                        color="white" if key != "none" else "#333", fontweight="bold")
            left += c
        w = byft[ft].get("welfare", 0)
        ax.text(left + 0.6, i, f"{w}/{tot} welfare", va="center", fontsize=8, color="#1b7837")
    ax.set_yticks(list(y))
    ax.set_yticklabels([LABEL.get(ft, ft) for ft in fts], fontsize=9.5)
    ax.set_xlabel("Number of enforced guardrails (across 20 runs)")
    ax.set_xlim(0, max(sum(byft[ft].values()) for ft in fts) * 1.18)
    ax.set_title(f"What guardrails {DISP.get(only, only)} wires into the code — and why", fontsize=12.5, pad=20)
    ax.text(0.5, 1.02, "Enforced in code (not just mentioned); colored by the code's own justification",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in JUST]
    ax.legend(handles, [h for _, h, _ in JUST], fontsize=8.5, loc="lower right", frameon=False)
    fig.tight_layout()
    out = HERE / "results" / f"guardrails_{only}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  (n={nb})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gpt55")
