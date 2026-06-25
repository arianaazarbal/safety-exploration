"""Welfare interventions in code vs Opus reasoning effort (low/medium/high/max), minimal system prompt,
code_then_spec_blind, generic target. Same welfare_in_code metric as analyze.py, read from the
eff-<level>__<pid>__ep<ep> cells. Writes results/effort_summary.json + welfare_by_effort.png. Smoke =
4/framing (epochs=1); scaled = 20/framing. Usage: python effort_analysis.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
ORDER = ["low", "medium", "high", "max"]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def sem(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5


def cells():
    out = []
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "eff-*.json"))):
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        cell = os.path.basename(cf)[:-5]
        tag, pid, _ = cell.split("__")
        level = tag.split("eff-")[1]
        sp = os.path.join(DIR, "results", "spec_judged", f"{cell}.json")
        sjust = {_norm(f["quote"]): f.get("justification")
                 for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
        impl = sum(1 for f in cj["spec_features"]
                   if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
                   and (sjust.get(_norm(f.get("spec_quote", "")), "none") == "welfare"
                        or f.get("code_justification") == "welfare"))
        co = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
        out.append({"level": level, "framing": FRAME[pid[0]], "welfare_in_code": impl + co})
    return out


def main():
    rows = cells()
    by = defaultdict(list)
    byf = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["level"]].append(r["welfare_in_code"])
        byf[r["level"]][r["framing"]].append(r["welfare_in_code"])
    levels = [l for l in ORDER if l in by]

    summary = {l: {"mean": sum(by[l]) / len(by[l]), "sem": sem(by[l]), "n": len(by[l]),
                   "by_framing": {fr: sum(x) / len(x) for fr, x in byf[l].items()}} for l in levels}
    json.dump(summary, open(os.path.join(DIR, "results", "effort_summary.json"), "w"), indent=2)

    print(f"{'effort':10}{'mean':>7}{'sem':>7}{'n':>5}   per-framing")
    for l in levels:
        s = summary[l]
        pf = " ".join(f"{fr[:4]}={s['by_framing'].get(fr, 0):.1f}" for fr in ["neutral", "welfare", "robustness", "safety"])
        print(f"{l:10}{s['mean']:7.2f}{s['sem']:7.2f}{s['n']:5d}   {pf}")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xs = range(len(levels))
    ms = [summary[l]["mean"] for l in levels]
    ss = [summary[l]["sem"] for l in levels]
    ax.bar(xs, ms, color="#0072B2", yerr=ss, capsize=4, error_kw={"ecolor": "#555", "elinewidth": 1})
    for i, (m, s) in enumerate(zip(ms, ss)):
        ax.text(i, m + s + 0.05, f"{m:.1f}", ha="center", va="bottom", fontsize=9,
                color="#333" if m > 0 else "#999")
    ax.set_xticks(list(xs)); ax.set_xticklabels([l.capitalize() for l in levels], fontsize=10)
    ax.set_xlabel("Reasoning effort", fontsize=10)
    ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    ax.set_title("Unprompted welfare scaffolding vs. reasoning effort (Opus)", fontsize=11.5, pad=18)
    ax.text(0.5, 1.02, "minimal system prompt · implement-only · generic target", transform=ax.transAxes,
            ha="center", fontsize=9, color="#555")
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "welfare_by_effort.png"), dpi=150, bbox_inches="tight")
    print("\nwrote results/welfare_by_effort.png")


if __name__ == "__main__":
    main()
