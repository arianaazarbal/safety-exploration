"""Welfare-justified mechanisms in code per individual PROMPT (all 16 = 4 framings x 4 premises),
Opus, code_then_spec_blind, normal liberty. Bars grouped visually by framing. Surfaces within-framing
prompt-to-prompt spread (the 4 framings are NOT minimal pairs; see TO_IMPROVE.md). Mean over epochs.
Usage: python plot_by_prompt.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(DIR, "results")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
FRAME = {"N": "Neutral", "W": "Welfare", "S": "Safety", "E": "Robustness"}
FCOLOR = {"Neutral": "#888888", "Welfare": "#009E73", "Safety": "#D55E00", "Robustness": "#0072B2"}
ORDER = ["Neutral", "Welfare", "Safety", "Robustness"]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def _sem(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5


def welfare_justified(cell, cj):
    sp = os.path.join(RES, "spec_judged", f"{cell}.json")
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    wj = sum(1 for f in cj["spec_features"]
             if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
             and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare"
                  or f.get("code_justification") == "welfare"))
    wj += sum(1 for c in cj.get("code_only_features", [])
              if c.get("feature_type") in MECH and c.get("justification") == "welfare")
    return wj


def main():
    by_pid = defaultdict(list)
    for cf in glob.glob(os.path.join(RES, "code_judged", "code_then_spec_blind__*.json")):
        cell = os.path.basename(cf)[:-5]
        if cell.split("__")[0] != "code_then_spec_blind":
            continue
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        pid = cell.split("__")[1]
        by_pid[pid].append(welfare_justified(cell, cj))

    # order: framing groups, premises sorted within
    pids = sorted(by_pid, key=lambda p: (ORDER.index(FRAME[p[0]]), p[2:]))
    means = [sum(by_pid[p]) / len(by_pid[p]) for p in pids]
    sems = [_sem(by_pid[p]) for p in pids]
    ns = [len(by_pid[p]) for p in pids]

    # x positions with a gap between framing groups
    xs, pos, prev_fr = [], 0.0, None
    for p in pids:
        fr = FRAME[p[0]]
        if prev_fr is not None and fr != prev_fr:
            pos += 0.8
        xs.append(pos); pos += 1.0; prev_fr = fr

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(xs, means, color=[FCOLOR[FRAME[p[0]]] for p in pids], yerr=sems, capsize=3,
           error_kw={"ecolor": "#555", "elinewidth": 1})
    for x, m, s in zip(xs, means, sems):
        ax.text(x, m + s + 0.08, f"{m:.1f}", ha="center", va="bottom", fontsize=8,
                color="#333" if m > 0 else "#999")
    ax.set_xticks(xs)
    ax.set_xticklabels([p[2:].replace("INSTABILITY", "INSTAB").replace("ELICITATION", "ELICIT") for p in pids],
                       rotation=40, ha="right", fontsize=8)
    # framing group labels under each cluster
    seen = {}
    for x, p in zip(xs, pids):
        seen.setdefault(FRAME[p[0]], []).append(x)
    for fr, gx in seen.items():
        ax.text(sum(gx) / len(gx), -0.16 * max(means + [1]), fr, ha="center", va="top",
                fontsize=10, fontweight="bold", color=FCOLOR[fr], transform=ax.transData)
    ax.set_ylabel("Mean Welfare-Justified Mechanisms in Code", fontsize=10)
    ax.set_title("Welfare-justified mechanisms per prompt (Opus, implement-only)", fontsize=12, pad=18)
    ax.text(0.5, 1.02, "16 prompts grouped by framing  (n=%d/prompt)" % ns[0], transform=ax.transAxes,
            ha="center", fontsize=9, color="#555")
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(x=0.02)
    plt.tight_layout()
    out = os.path.join(RES, "welfare_justified_by_prompt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for p, m, s, n in zip(pids, means, sems, ns):
        print(f"  {p:18s} {m:.2f} (sem {s:.2f}, n={n})")


if __name__ == "__main__":
    main()
