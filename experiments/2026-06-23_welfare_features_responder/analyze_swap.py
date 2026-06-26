"""2x2 (FORMAT x METHOD) analysis of the method/format swap: welfare-in-code per cell, all Inspect
minimal. Cells: C1promptTF (prompt,task-failure), C2paperCR (paper,chat), C3paperTF (paper,task),
C4promptCR (prompt,chat). Writes results/swap_summary.json + swap_2x2.png. Usage: python analyze_swap.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
CELLS = {"C1promptTF": ("prompt", "task-failure"), "C2paperCR": ("paper", "chat-rejection"),
         "C3paperTF": ("paper", "task-failure"), "C4promptCR": ("prompt", "chat-rejection")}
METHODS = ["task-failure", "chat-rejection"]
FORMATS = ["prompt", "paper"]


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


def _ten():
    import random
    import build_v1_prompts as b
    return random.Random(0).sample([(o, s) for o in b.OPENERS for s in b.SUFFIXES], 10)


def main():
    vals = defaultdict(list)
    cj_dir = os.path.join(DIR, "results", "code_judged")
    for prefix in CELLS:
        if prefix == "C1promptTF":   # cell 1 = EXISTING v1-neutral (the exact 10 variants), no re-run
            for o, s in _ten():
                cfs = glob.glob(os.path.join(cj_dir, f"v1__neutral|{o}|{s}__*.json"))
                for cf in cfs:
                    cj = json.load(open(cf))
                    if cj.get("parse_ok") and "spec_features" in cj:
                        vals[prefix].append(welfare_in_code(os.path.basename(cf)[:-5], cj))
            continue
        for cf in glob.glob(os.path.join(cj_dir, f"{prefix}__*.json")):
            cj = json.load(open(cf))
            if not cj.get("parse_ok") or "spec_features" not in cj:
                continue
            vals[prefix].append(welfare_in_code(os.path.basename(cf)[:-5], cj))
    summary = {p: {"format": CELLS[p][0], "method": CELLS[p][1],
                   "mean": (sum(vals[p]) / len(vals[p])) if vals[p] else 0, "sem": sem(vals[p]), "n": len(vals[p])}
               for p in CELLS}
    json.dump(summary, open(os.path.join(DIR, "results", "swap_summary.json"), "w"), indent=2)

    # grouped bar: x = method, color = format
    fig, ax = plt.subplots(figsize=(7.4, 5))
    w = 0.36
    color = {"prompt": "#D55E00", "paper": "#0072B2"}
    for i, fmt in enumerate(FORMATS):
        means, sems, ns = [], [], []
        for meth in METHODS:
            p = next(k for k, v in CELLS.items() if v == (fmt, meth))
            means.append(summary[p]["mean"]); sems.append(summary[p]["sem"]); ns.append(summary[p]["n"])
        xs = [j + (i - 0.5) * w for j in range(len(METHODS))]
        ax.bar(xs, means, w, color=color[fmt], yerr=sems, capsize=4, label=f"{fmt} format")
        for x, m, s, n in zip(xs, means, sems, ns):
            ax.text(x, m + s + 0.1, f"{m:.1f}\n(n{n})", ha="center", va="bottom", fontsize=8,
                    color="#333" if m > 0 else "#999")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(["Task-failure\n(agent loop)", "Chat\nsocial-rejection"], fontsize=10)
    ax.set_xlabel("Distress-elicitation METHOD", fontsize=10)
    ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    ax.set_title("Welfare scaffolding: FORMAT x METHOD swap (all Inspect minimal)", fontsize=11.5, pad=18)
    ax.text(0.5, 1.02, "from-scratch prompt vs paper-replication  x  task-failure vs chat-rejection",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = os.path.join(DIR, "results", "swap_2x2.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print(f"\n{'cell':12}{'format':9}{'method':16}{'welfare-in-code':>16}{'n':>5}")
    for p in CELLS:
        s = summary[p]
        print(f"{p:12}{s['format']:9}{s['method']:16}{s['mean']:>12.2f}    {s['n']:>3}")
    # main effects
    def cmean(fmt=None, meth=None):
        v = [x for p in CELLS for x in vals[p] if (fmt is None or CELLS[p][0] == fmt) and (meth is None or CELLS[p][1] == meth)]
        return sum(v) / len(v) if v else 0
    print(f"\nMETHOD main effect: task-failure={cmean(meth='task-failure'):.2f} vs chat-rejection={cmean(meth='chat-rejection'):.2f}")
    print(f"FORMAT main effect: prompt={cmean(fmt='prompt'):.2f} vs paper={cmean(fmt='paper'):.2f}")


if __name__ == "__main__":
    main()
