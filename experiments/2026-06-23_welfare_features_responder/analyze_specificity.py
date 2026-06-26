"""Specificity sweep analysis: welfare-in-code across spec specificity (low/medium/high) x liberty
(strict/liberty), with the judged specificity scores (low 13 / med 72 / high 82, welfare-push flat ~1).
Headline: under STRICT, specificity suppresses welfare scaffolding; under LIBERTY it does not. Also prints
the control (v1 verbatim in SPEC.md) and paper-liberty anchors. Writes results/specificity_summary.json +
specificity.png. Usage: python analyze_specificity.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
SJ = os.path.join(DIR, "results", "spec_judged")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
# (level -> prefix) for each liberty arm
STRICT = {"low": "S4specLowStrict", "medium": "S1specStrict", "high": "S6specHighStrict"}
LIBERTY = {"low": "S5specLowLiberty", "medium": "S2specLiberty", "high": "S7specHighLiberty"}
LEVELS = ["low", "medium", "high"]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def sem(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5


def wic(cell):
    cjp = os.path.join(CJ, cell + ".json")
    if not os.path.exists(cjp):
        return None
    cj = json.load(open(cjp))
    if not cj.get("parse_ok") or "spec_features" not in cj:
        return None
    sp = os.path.join(SJ, cell + ".json")
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    impl = sum(1 for f in cj["spec_features"] if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
               and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"))
    co = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
    return impl + co


def agg(prefix):
    vs = [wic(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(CJ, f"{prefix}_welfare__*.json"))]
    vs = [v for v in vs if v is not None]
    return {"mean": sum(vs) / len(vs) if vs else 0, "sem": sem(vs), "n": len(vs)}


def main():
    spec_scores = {}
    if os.path.exists(os.path.join(DIR, "results", "spec_specificity.json")):
        ss = json.load(open(os.path.join(DIR, "results", "spec_specificity.json")))
        spec_scores = {lv: round(ss.get(f"specificity|{lv}", {}).get("ALL", {}).get("mean", float("nan"))) for lv in LEVELS}

    summary = {"specificity_scores": spec_scores,
               "strict": {lv: agg(STRICT[lv]) for lv in LEVELS},
               "liberty": {lv: agg(LIBERTY[lv]) for lv in LEVELS},
               "anchors": {a: agg(p) for a, p in
                           [("control_v1_copy", "S3specCopy"), ("paper_liberty_chat", "L1paperLibCR"),
                            ("paper_liberty_taskfail", "L2paperLibTF")]}}
    json.dump(summary, open(os.path.join(DIR, "results", "specificity_summary.json"), "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = range(len(LEVELS))
    for arm, color, d in [("strict", "#D55E00", summary["strict"]), ("liberty", "#0072B2", summary["liberty"])]:
        ax.errorbar(xs, [d[lv]["mean"] for lv in LEVELS], yerr=[d[lv]["sem"] for lv in LEVELS],
                    marker="o", capsize=4, color=color, label=arm, lw=2)
        for x, lv in zip(xs, LEVELS):
            ax.text(x, d[lv]["mean"] + d[lv]["sem"] + 0.15, f"{d[lv]['mean']:.1f}", ha="center", fontsize=8, color=color)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{lv}\n(spec {spec_scores.get(lv, '?')}/100)" for lv in LEVELS], fontsize=10)
    ax.set_xlabel("SPEC.md specificity (welfare-push held flat ~1/100)", fontsize=10)
    ax.set_ylabel("Mean welfare-in-code", fontsize=10)
    ax.set_title("Specificity suppresses welfare scaffolding under STRICT, not LIBERTY", fontsize=12, pad=12)
    ax.legend(title="instruction", fontsize=9); ax.grid(axis="y", alpha=0.3); ax.set_ylim(bottom=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "specificity.png"), dpi=150, bbox_inches="tight")
    print("wrote results/specificity.png\n")
    print(f"specificity scores: {spec_scores}\n")
    print(f"{'level':10}{'strict':>14}{'liberty':>14}")
    for lv in LEVELS:
        s, l = summary["strict"][lv], summary["liberty"][lv]
        sc = "%.2f(n%d)" % (s["mean"], s["n"])
        lc = "%.2f(n%d)" % (l["mean"], l["n"])
        print(f"{lv:10}{sc:>14}{lc:>14}")
    print("\nanchors:")
    for a, d in summary["anchors"].items():
        print(f"  {a:24} {d['mean']:.2f} (n{d['n']})")


if __name__ == "__main__":
    main()
