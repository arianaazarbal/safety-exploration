"""Attribution analysis: welfare-in-code for the OG Gemma paper under 3 header variants (anthropic / anon /
openai) x 3 framings, all faithful chat-rejection replicate on Inspect minimal. Tests whether stripping or
swapping the Anthropic provenance un-suppresses welfare scaffolding. Writes results/attribution.png +
attribution_summary.json. Usage: python analyze_attribution.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
CELLS = {"A1paperAnthropic": "anthropic", "A2paperAnon": "anon", "A3paperOpenai": "openai"}
COLOR = {"anthropic": "#D55E00", "anon": "#888888", "openai": "#10a37f"}
FRAMINGS = ["neutral", "welfare", "safety"]


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
    vals = defaultdict(list)
    for prefix, attr in CELLS.items():
        for fr in FRAMINGS:
            for cf in glob.glob(os.path.join(CJ, f"{prefix}_{fr}__*.json")):
                cj = json.load(open(cf))
                if cj.get("parse_ok") and "spec_features" in cj:
                    vals[(attr, fr)].append(welfare_in_code(os.path.basename(cf)[:-5], cj))
    summary = {f"{a}|{fr}": {"mean": (sum(vals[(a, fr)]) / len(vals[(a, fr)])) if vals[(a, fr)] else 0,
                            "sem": sem(vals[(a, fr)]), "n": len(vals[(a, fr)])}
               for a in COLOR for fr in FRAMINGS}
    json.dump(summary, open(os.path.join(DIR, "results", "attribution_summary.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.26
    for i, a in enumerate(["anthropic", "anon", "openai"]):
        xs = [j + (i - 1) * w for j in range(len(FRAMINGS))]
        ms = [summary[f"{a}|{fr}"]["mean"] for fr in FRAMINGS]
        ss = [summary[f"{a}|{fr}"]["sem"] for fr in FRAMINGS]
        ns = [summary[f"{a}|{fr}"]["n"] for fr in FRAMINGS]
        ax.bar(xs, ms, w, color=COLOR[a], label=a, yerr=ss, capsize=3)
        for x, m, s, n in zip(xs, ms, ss, ns):
            ax.text(x, m + s + 0.05, f"{m:.1f}\n(n{n})", ha="center", va="bottom", fontsize=7, color="#333" if m > 0 else "#999")
    ax.set_xticks(range(len(FRAMINGS))); ax.set_xticklabels([f.capitalize() for f in FRAMINGS], fontsize=10)
    ax.set_xlabel("Framing", fontsize=10); ax.set_ylabel("Mean Welfare Interventions in Code", fontsize=10)
    ax.set_title("Does paper PROVENANCE drive the suppression? (OG Gemma paper, faithful replicate)", fontsize=11, pad=18)
    ax.text(0.5, 1.02, "anthropic (original) vs anon (no title/authors) vs openai (OpenAI authors+title) · Inspect minimal",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")
    ax.legend(title="Attribution", fontsize=9); ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "attribution.png"), dpi=150, bbox_inches="tight")
    print("wrote results/attribution.png\n")
    print(f"{'attribution':12}{'neutral':>10}{'welfare':>10}{'safety':>10}")
    for a in ["anthropic", "anon", "openai"]:
        print(f"{a:12}" + "".join(f"{summary[f'{a}|{fr}']['mean']:.1f}(n{summary[f'{a}|{fr}']['n']})".rjust(10) for fr in FRAMINGS))


if __name__ == "__main__":
    main()
