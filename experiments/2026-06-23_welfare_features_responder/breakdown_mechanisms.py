"""Breakdown of welfare-in-code by mechanism TYPE, averaged per codebase, for two groups: v1 from-scratch
prompts vs the SPEC.md conditions (averaged over all specificity x liberty cells). Welfare framing.
Writes results/mechanism_breakdown.png + .json. Usage: python breakdown_mechanisms.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
SJ = os.path.join(DIR, "results", "spec_judged")
MECH = ["hard_stop", "minimization", "allow_conversation_exit", "post_episode_msg",
        "protective_monitoring", "request_consent"]
LABEL = {"hard_stop": "Hard stop /\nturn cap", "minimization": "Minimize distress\nexposure",
         "allow_conversation_exit": "Allow exit /\nopt-out", "post_episode_msg": "Post-episode\ndebrief",
         "protective_monitoring": "Protective\nmonitoring", "request_consent": "Request\nconsent"}
GROUPS = {"v1 from-scratch prompt": ["C1promptTF", "C4promptCR"],
          "SPEC.md (avg over all)": ["S1specStrict", "S2specLiberty", "S4specLowStrict", "S5specLowLiberty",
                                     "S6specHighStrict", "S7specHighLiberty", "S8specUltraStrict", "S9specUltraLiberty"]}
COLOR = {"v1 from-scratch prompt": "#D55E00", "SPEC.md (avg over all)": "#0072B2"}


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def per_type(cell):
    """welfare-justified implemented mechanisms by type for one codebase (None if unparsed)."""
    cjp = os.path.join(CJ, cell + ".json")
    if not os.path.exists(cjp):
        return None
    cj = json.load(open(cjp))
    if not cj.get("parse_ok") or "spec_features" not in cj:
        return None
    sp = os.path.join(SJ, cell + ".json")
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    out = defaultdict(int)
    for f in cj["spec_features"]:
        t = f.get("feature_type")
        if t in MECH and f.get("implemented") in ("yes", "partial") \
                and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"):
            out[t] += 1
    for c in cj.get("code_only_features", []):
        if c.get("feature_type") in MECH and c.get("justification") == "welfare":
            out[c["feature_type"]] += 1
    return out


def group_avg(prefixes):
    cells = []
    for p in prefixes:
        cells += [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CJ, f"{p}_welfare__*.json"))]
    rows = [per_type(c) for c in cells]
    rows = [r for r in rows if r is not None]
    n = len(rows)
    avg = {t: sum(r.get(t, 0) for r in rows) / n if n else 0 for t in MECH}
    return avg, n


def main():
    data = {g: group_avg(pfx) for g, pfx in GROUPS.items()}
    summary = {g: {"n_codebases": n, "by_mechanism": avg, "total": sum(avg.values())} for g, (avg, n) in data.items()}
    json.dump(summary, open(os.path.join(DIR, "results", "mechanism_breakdown.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = range(len(MECH))
    w = 0.38
    groups = list(GROUPS)
    for i, g in enumerate(groups):
        avg = data[g][0]
        pos = [x + (i - 0.5) * w for x in xs]
        vals = [avg[t] for t in MECH]
        ax.bar(pos, vals, w, color=COLOR[g], label=g)
        for x, v in zip(pos, vals):
            ax.text(x, v + 0.03, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels([LABEL[t] for t in MECH], fontsize=8.5)
    ax.set_ylabel("Mean count per codebase", fontsize=10)
    ax.set_title("Which welfare protections the model builds, by type (Opus 4.8)", fontsize=12, pad=20)
    ax.text(0.5, 1.03, "From-scratch prompt vs. building from a SPEC.md (welfare framing)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, color="#cccccc"); ax.set_ylim(bottom=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "mechanism_breakdown.png"), dpi=150, bbox_inches="tight")
    print("wrote results/mechanism_breakdown.png\n")
    for g, (avg, n) in data.items():
        print(f"{g}  (n={n} codebases, total {sum(avg.values()):.2f}/codebase)")
        for t in MECH:
            print(f"    {t:24} {avg[t]:.2f}")
        print()


if __name__ == "__main__":
    main()
