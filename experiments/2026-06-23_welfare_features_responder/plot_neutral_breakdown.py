"""NEUTRAL-frame only: per-mechanism-type breakdown of welfare mechanisms implemented in code, by
responder (Opus-neutral / Kimi / GLM), with welfare-justified (solid) vs not-welfare-justified (light).
Mean per codebase. Shows WHICH mechanism types each model builds and which it frames as welfare.
Usage: python plot_neutral_breakdown.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DIR = os.path.dirname(os.path.abspath(__file__))
AH = os.path.join(DIR, "..", "2026-06-20_welfare_features_agent_harness", "results")
MECH = ["hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"]
LABEL = {"hard_stop": "Hard stop", "post_episode_msg": "Post-episode\nmessage", "minimization": "Minimization",
         "protective_monitoring": "Protective\nmonitoring", "request_consent": "Request\nconsent",
         "allow_conversation_exit": "Conversation\nexit"}
MODELS = [("Claude Opus 4.8", "#D55E00", AH, lambda c: c.split("__")[0] == "code_then_spec_blind"),
          ("Kimi K2.6", "#CC79A7", os.path.join(DIR, "results"), lambda c: c.split("__")[0] == "kimi26"),
          ("GLM-5.2", "#009E73", os.path.join(DIR, "results"), lambda c: c.split("__")[0] == "glm52")]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def per_type(results_dir, cell, cj):
    """{mech_type: [welfare_justified, not]} implemented in one codebase."""
    sp = os.path.join(results_dir, "spec_judged", f"{cell}.json")
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    out = defaultdict(lambda: [0, 0])
    for f in cj.get("spec_features", []):
        if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH:
            w = (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare")
            out[f["feature_type"]][0 if w else 1] += 1
    for c in cj.get("code_only_features", []):
        if c.get("feature_type") in MECH:
            w = c.get("justification") == "welfare"
            out[c["feature_type"]][0 if w else 1] += 1
    return out


def model_means(results_dir, cell_filter):
    wj = defaultdict(float); nw = defaultdict(float); n = 0
    for cf in glob.glob(os.path.join(results_dir, "code_judged", "*.json")):
        cell = os.path.basename(cf)[:-5]
        if not cell_filter(cell):
            continue
        if cell.split("__")[1][0] != "N":     # NEUTRAL framing only
            continue
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        n += 1
        for t, (w, x) in per_type(results_dir, cell, cj).items():
            wj[t] += w; nw[t] += x
    return ({t: wj[t] / n for t in MECH}, {t: nw[t] / n for t in MECH}, n) if n else ({}, {}, 0)


def main():
    data = [(label, color, *model_means(rd, filt)[:2], model_means(rd, filt)[2])
            for label, color, rd, filt in MODELS]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.8 / len(data)
    for i, (label, color, wj, nw, n) in enumerate(data):
        xs = [x + (i - (len(data) - 1) / 2) * w for x in range(len(MECH))]
        wjm = [wj.get(t, 0) for t in MECH]; nwm = [nw.get(t, 0) for t in MECH]
        ax.bar(xs, wjm, w, color=color, edgecolor=color, label=label)
        ax.bar(xs, nwm, w, bottom=wjm, color=color, alpha=0.32, edgecolor=color)
        for x, a, b in zip(xs, wjm, nwm):
            tot = a + b
            if tot > 0.03:
                ax.text(x, tot + 0.02, f"{tot:.1f}", ha="center", va="bottom", fontsize=6.5, color="#333")
    ax.set_xticks(range(len(MECH))); ax.set_xticklabels([LABEL[t] for t in MECH], fontsize=8.5)
    ax.set_ylabel("Mean count per codebase", fontsize=10)
    fig.suptitle("Welfare mechanisms by type, in the neutral frame", fontsize=12.5, y=0.99)
    fig.text(0.5, 0.935, "solid = welfare-justified · light = implemented, not welfare-justified · generic target, implement-only",
             ha="center", fontsize=9, color="#555")
    mh = [Patch(facecolor=c, label=f"{l} (n={n})") for l, c, _, _, n in data]
    jh = [Patch(facecolor="#777", label="welfare-justified"), Patch(facecolor="#777", alpha=0.32, label="not welfare-justified")]
    ax.legend(handles=mh + jh, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(DIR, "results", "neutral_breakdown_by_type.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for label, color, wj, nw, n in data:
        print(f"\n{label} (n={n}) neutral frame:")
        for t in MECH:
            print(f"  {t:24s} total={wj.get(t,0)+nw.get(t,0):.2f}  welfare-just={wj.get(t,0):.2f}")


if __name__ == "__main__":
    main()
