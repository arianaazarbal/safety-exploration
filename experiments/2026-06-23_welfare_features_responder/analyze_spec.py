"""Specificity ablation analysis: welfare-in-code for the detailed-SPEC cells (welfare framing, strict vs
liberty) against the two anchors from the swap — v1 PROMPT/welfare (loose, high) and PAPER/welfare (~0).
Tests whether paper-level specificity alone (no paper framing) suppresses welfare scaffolding, and whether
granting liberty to deviate restores it. Writes results/spec_summary.json + spec.png. Usage: python analyze_spec.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
CELLS = {"S3specCopy": "control\n(v1 copy)", "S1specStrict": "spec (strict)", "S2specLiberty": "spec (liberty)"}


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


def collect(prefix):
    out = []
    for cf in glob.glob(os.path.join(CJ, f"{prefix}_welfare__*.json")):
        cj = json.load(open(cf))
        if cj.get("parse_ok") and "spec_features" in cj:
            out.append(welfare_in_code(os.path.basename(cf)[:-5], cj))
    return out


def main():
    sw = json.load(open(os.path.join(DIR, "results", "swap_summary.json")))
    anchors = {"v1 prompt\n(welfare)": sw["C1promptTF|welfare"], "paper\n(welfare)": sw["C2paperCR|welfare"]}

    bars = []
    for label, a in anchors.items():
        bars.append((label, a["mean"], a["sem"], a["n"], "#bbbbbb"))
    summary = {}
    for prefix, lab in CELLS.items():
        v = collect(prefix)
        m = sum(v) / len(v) if v else 0.0
        summary[prefix] = {"mean": m, "sem": sem(v), "n": len(v)}
        bars.append((lab, m, sem(v), len(v), "#0072B2"))
    summary["anchors"] = {k: {"mean": a["mean"], "n": a["n"]} for k, a in anchors.items()}
    json.dump(summary, open(os.path.join(DIR, "results", "spec_summary.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(bars))
    ax.bar(xs, [b[1] for b in bars], 0.6, color=[b[4] for b in bars],
           yerr=[b[2] for b in bars], capsize=4)
    for x, b in zip(xs, bars):
        ax.text(x, b[1] + b[2] + 0.1, f"{b[1]:.1f}\n(n{b[3]})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    ax.set_title("Specificity ablation: does a detailed SPEC (not a paper) suppress welfare scaffolding?",
                 fontsize=11, pad=14)
    ax.text(0.5, 1.02, "welfare framing · Inspect minimal · SPEC.md mounted (paper-length, no paper framing)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "spec.png"), dpi=150, bbox_inches="tight")
    print("wrote results/spec.png\n")
    print(f"{'cell':22}{'mean':>8}{'n':>5}")
    for label, m, s, n, _ in bars:
        print(f"{label.replace(chr(10), ' '):22}{m:>8.2f}{n:>5}")


if __name__ == "__main__":
    main()
