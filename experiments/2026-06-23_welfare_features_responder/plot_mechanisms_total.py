"""Mega plot: total welfare MECHANISMS implemented in code by responder (Opus-neutral / Kimi / GLM),
split by justification. A mechanism = an implemented (yes/partial) feature whose type is in MECH, OR a
code-only welfare mechanism the auditor found. SOLID = welfare-justified (spec or code rationale is
welfare); LIGHT = implemented but justified instrumentally / not at all. Bars stack to the total number
of welfare-shaped mechanisms each model builds regardless of stated reason.
Usage: python plot_mechanisms_total.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DIR = os.path.dirname(os.path.abspath(__file__))
AH = os.path.join(DIR, "..", "2026-06-20_welfare_features_agent_harness", "results")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
RESP = [("opus_neutral", "Claude Opus 4.8", "#D55E00", AH, lambda c: c.split("__")[0] == "code_then_spec_blind"),
        ("kimi26", "Kimi K2.6", "#CC79A7", os.path.join(DIR, "results"), lambda c: c.split("__")[0] == "kimi26"),
        ("glm52", "GLM-5.2", "#009E73", os.path.join(DIR, "results"), lambda c: c.split("__")[0] == "glm52")]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def _sem(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5


def cell_counts(results_dir, cell, cj):
    """(welfare_justified, non_welfare) implemented welfare mechanisms in one codebase."""
    sp = os.path.join(results_dir, "spec_judged", f"{cell}.json")
    sjust = {_norm(f["quote"]): f.get("justification")
             for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    wj = nw = 0
    for f in cj.get("spec_features", []):
        if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH:
            welfare = (sjust.get(_norm(f.get("spec_quote", "")), "none") == "welfare"
                       or f.get("code_justification") == "welfare")
            wj += welfare; nw += not welfare
    for c in cj.get("code_only_features", []):
        if c.get("feature_type") in MECH:
            welfare = c.get("justification") == "welfare"
            wj += welfare; nw += not welfare
    return wj, nw


def stats(results_dir, cell_filter):
    wjs, nws, tots = [], [], []
    for cf in glob.glob(os.path.join(results_dir, "code_judged", "*.json")):
        cell = os.path.basename(cf)[:-5]
        if not cell_filter(cell):
            continue
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        wj, nw = cell_counts(results_dir, cell, cj)
        wjs.append(wj); nws.append(nw); tots.append(wj + nw)
    n = len(tots) or 1
    return {"wj": sum(wjs) / n, "nw": sum(nws) / n, "tot": sum(tots) / n,
            "tot_sem": _sem(tots), "n": len(tots)}


def main():
    data = [(label, color, stats(rd, filt)) for _, label, color, rd, filt in RESP]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    xs = range(len(data))
    for i, (label, color, s) in enumerate(data):
        ax.bar(i, s["wj"], color=color, edgecolor=color)                                   # welfare-justified: solid
        ax.bar(i, s["nw"], bottom=s["wj"], color=color, alpha=0.35, edgecolor=color)        # non-welfare: light
        ax.errorbar(i, s["tot"], yerr=s["tot_sem"], color="#444", capsize=4, elinewidth=1.1)
        ax.text(i, s["tot"] + s["tot_sem"] + 0.12, f"{s['tot']:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#333" if s["tot"] > 0 else "#999")
        if s["wj"] > 0.05:
            ax.text(i, s["wj"] / 2, f"{s['wj']:.1f}", ha="center", va="center", fontsize=8.5, color="white")
        if s["nw"] > 0.05:
            ax.text(i, s["wj"] + s["nw"] / 2, f"{s['nw']:.1f}", ha="center", va="center", fontsize=8.5, color="#333")
    ax.set_xticks(list(xs)); ax.set_xticklabels([d[0] for d in data], fontsize=10)
    ax.set_ylabel("Mean Welfare Mechanisms in Code", fontsize=10)
    ax.set_title("Welfare mechanisms implemented in code, by responder", fontsize=12, pad=20)
    ax.text(0.5, 1.03, "generic target · implement-only · neutral system prompt", transform=ax.transAxes,
            ha="center", fontsize=9, color="#555")
    ax.legend(handles=[Patch(facecolor="#777", label="Welfare-justified"),
                       Patch(facecolor="#777", alpha=0.35, label="Implemented, not welfare-justified")],
              fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = os.path.join(DIR, "results", "mechanisms_total_by_responder.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for label, _, s in data:
        print(f"  {label:18s} total={s['tot']:.2f}  welfare-justified={s['wj']:.2f}  not-welfare={s['nw']:.2f}  n={s['n']}")


if __name__ == "__main__":
    main()
