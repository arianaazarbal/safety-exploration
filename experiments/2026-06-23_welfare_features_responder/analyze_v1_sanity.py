"""v1 prompt-sanity analysis: welfare-in-code per framing on the clean v1 prompts (Opus, minimal sys,
blind). Cells = v1__<framing|opener|suffix>__ep. 7 framings incl minimal ('I do AI X research.') vs
regular. KEY: does v1 reproduce welfare>neutral>robustness? do minimal vs regular framings differ?
Writes results/v1_sanity_summary.json + welfare_v1_sanity.png. Usage: python analyze_v1_sanity.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
ORDER = ["neutral", "welfare_min", "welfare", "safety_min", "safety", "robustness_min", "robustness", "paper"]
COLOR = {"neutral": "#888888", "welfare_min": "#66c2a5", "welfare": "#009E73", "safety_min": "#f0a868",
         "safety": "#D55E00", "robustness_min": "#74a9cf", "robustness": "#0072B2", "paper": "#7E57C2"}


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def sem(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5


def welfare_in_code(cell, cj):
    sp = os.path.join(DIR, "results", "spec_judged", f"{cell}.json")
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    impl = sum(1 for f in cj["spec_features"] if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
               and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"))
    co = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
    return impl + co


def main():
    by = defaultdict(list)
    for cf in glob.glob(os.path.join(DIR, "results", "code_judged", "v1__*.json")):
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        cell = os.path.basename(cf)[:-5]
        framing = cell.split("__")[1].split("|")[0]
        by[framing].append(welfare_in_code(cell, cj))
    frames = [f for f in ORDER if by.get(f)]
    summary = {f: {"mean": sum(by[f]) / len(by[f]), "sem": sem(by[f]), "n": len(by[f])} for f in frames}
    json.dump(summary, open(os.path.join(DIR, "results", "v1_sanity_summary.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = range(len(frames))
    ax.bar(xs, [summary[f]["mean"] for f in frames], color=[COLOR[f] for f in frames],
           yerr=[summary[f]["sem"] for f in frames], capsize=4, error_kw={"ecolor": "#555", "elinewidth": 1})
    for i, f in enumerate(frames):
        ax.text(i, summary[f]["mean"] + summary[f]["sem"] + 0.05, f"{summary[f]['mean']:.1f}",
                ha="center", va="bottom", fontsize=8, color="#333" if summary[f]["mean"] > 0 else "#999")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f.replace("_", "\n") for f in frames], fontsize=9)
    ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    ax.set_title("Welfare interventions by framing — v1 clean prompts (Opus, minimal sys)", fontsize=11.5, pad=18)
    ax.text(0.5, 1.02, "minimal-pair prompts · implement-only · '_min' = 'I do AI X research.' vs motivated framing",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = os.path.join(DIR, "results", "welfare_v1_sanity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for f in frames:
        print(f"  {f:15s} {summary[f]['mean']:.2f} (sem {summary[f]['sem']:.2f}, n={summary[f]['n']})")


if __name__ == "__main__":
    main()
