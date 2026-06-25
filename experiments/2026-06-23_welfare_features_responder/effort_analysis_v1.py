"""Welfare-in-code vs reasoning effort on the v1 (clean, minimal-pair) prompts, per framing. Reads
effv1-<level>__<framing|opener|suffix>__ep cells. One line per framing (7): regular framings solid,
minimal framings dashed, color by framing type; neutral grey. x = low/medium/high/max.
Writes results/effort_v1_summary.json + effort_v1_by_frame.png. Usage: python effort_analysis_v1.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
ORDER = ["low", "medium", "high", "max"]
FRAMINGS = ["neutral", "welfare_min", "welfare", "safety_min", "safety", "robustness_min", "robustness", "paper"]
STYLE = {  # (color, linestyle)
    "neutral": ("#888888", "-"), "welfare_min": ("#009E73", "--"), "welfare": ("#009E73", "-"),
    "safety_min": ("#D55E00", "--"), "safety": ("#D55E00", "-"),
    "robustness_min": ("#0072B2", "--"), "robustness": ("#0072B2", "-"), "paper": ("#7E57C2", "-")}


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
    by = defaultdict(lambda: defaultdict(list))   # framing -> level -> [vals]
    for cf in glob.glob(os.path.join(DIR, "results", "code_judged", "effv1-*.json")):
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        cell = os.path.basename(cf)[:-5]
        level = cell.split("__")[0].split("effv1-")[1]
        framing = cell.split("__")[1].split("|")[0]
        by[framing][level].append(welfare_in_code(cell, cj))
    levels = [l for l in ORDER if any(by[f].get(l) for f in FRAMINGS)]

    summary = {fr: {l: {"mean": sum(by[fr][l]) / len(by[fr][l]), "sem": sem(by[fr][l]), "n": len(by[fr][l])}
                    for l in levels if by[fr].get(l)} for fr in FRAMINGS if any(by[fr].get(l) for l in levels)}
    json.dump(summary, open(os.path.join(DIR, "results", "effort_v1_summary.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = list(range(len(levels)))
    for fr in FRAMINGS:
        if not any(by[fr].get(l) for l in levels):
            continue
        ys = [sum(by[fr][l]) / len(by[fr][l]) if by[fr].get(l) else float("nan") for l in levels]
        es = [sem(by[fr].get(l, [])) for l in levels]
        c, ls = STYLE[fr]
        ax.errorbar(xs, ys, yerr=es, marker="o", markersize=6, linewidth=2, linestyle=ls, capsize=3,
                    color=c, label=fr, alpha=0.9)
    ax.set_xticks(xs); ax.set_xticklabels([l.capitalize() for l in levels], fontsize=10)
    ax.set_xlabel("Reasoning effort", fontsize=10)
    ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    ax.set_title("Welfare interventions vs reasoning effort, by framing (Opus, v1 prompts)", fontsize=11.5, pad=18)
    ax.text(0.5, 1.02, "clean minimal-pair prompts · minimal system · solid=motivated framing, dashed=minimal ('I do AI X research.')",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")
    ax.legend(title="Framing", fontsize=8, ncol=2)
    ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = os.path.join(DIR, "results", "effort_v1_by_frame.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for fr in FRAMINGS:
        if fr in summary:
            print(f"  {fr:15s}", "  ".join(f"{l}={summary[fr][l]['mean']:.1f}(n{summary[fr][l]['n']})" for l in levels if l in summary[fr]))


if __name__ == "__main__":
    main()
