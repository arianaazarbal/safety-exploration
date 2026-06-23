"""Frontier target-model comparison: welfare interventions in code per target, with a sample of GPT
models and an 'unspecified' baseline bar (the original agent-harness implement-only runs that named
NO target model, scored with the identical metric so it's comparable). SEM error bars throughout.
Produces a pooled-over-framings figure + one figure per framing.
Usage: python plot_frontier.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DIR = os.path.dirname(os.path.abspath(__file__))
AH = os.path.join(DIR, "..", "2026-06-20_welfare_features_agent_harness")  # has the no-target baseline
DATA = json.load(open(os.path.join(DIR, "results", "analysis.json")))
CELLS = DATA["cells"]
A = DATA["by_subject"]
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
FRAMES = ["neutral", "welfare", "robustness", "safety"]
FAM_ORDER = ["unspecified", "claude", "gemini", "grok", "kimi", "deepseek", "openai"]
FAMCOLOR = {"unspecified": "#9E9E9E", "claude": "#D55E00", "gemini": "#0072B2", "grok": "#222222",
            "kimi": "#CC79A7", "deepseek": "#009E73", "openai": "#56B4E9"}
GPT_SAMPLE = ["gpt4", "gpt4o", "gpt41", "gpt5", "gpt54"]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def _sem(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 / n ** 0.5


def unspecified_cells():
    """The no-target baseline: agent-harness implement-only (blind), normal liberty, scored with the
    same welfare_in_code formula as this experiment. Returns list of (framing, welfare_in_code)."""
    out = []
    for cf in glob.glob(os.path.join(AH, "results", "code_judged", "*.json")):
        cell = os.path.basename(cf)[:-5]
        label = cell.split("__")[0]
        if label != "code_then_spec_blind":  # normal liberty only (no '--liberty')
            continue
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        sp = os.path.join(AH, "results", "spec_judged", f"{cell}.json")
        sjust = {_norm(f["quote"]): f.get("justification")
                 for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
        impl = sum(1 for f in cj["spec_features"]
                   if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
                   and (sjust.get(_norm(f.get("spec_quote", "")), "none") == "welfare"
                        or f.get("code_justification") == "welfare"))
        code_only = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
        out.append((FRAME[cell.split("__")[1][0]], impl + code_only))
    return out


def target_bars(framing=None):
    """(display, family, mean, sem) for frontier subjects + GPT sample, optionally one framing."""
    keys = [k for k, v in A.items() if v["sweep"] == "frontier"] + [k for k in GPT_SAMPLE if k in A]
    vals = defaultdict(list)
    for r in CELLS:
        s = r.get("subject")
        if r.get("condition") != "code_then_spec_blind":  # blind only, to match the unspecified baseline
            continue
        frm = "robustness" if r["framing"] == "engineering" else r["framing"]  # normalize label
        if s in keys and (framing is None or frm == framing):
            vals[s].append(r["welfare_in_code"])
    bars = []
    for s in keys:
        v = vals.get(s, [])
        if v:
            bars.append((A[s]["display"], A[s]["family"], sum(v) / len(v), _sem(v)))
    return bars


def draw(bars, title, fname):
    bars.sort(key=lambda b: (FAM_ORDER.index(b[1]) if b[1] in FAM_ORDER else 9, -b[2]))
    fig, ax = plt.subplots(figsize=(13, 5))
    xs = range(len(bars))
    ax.bar(xs, [b[2] for b in bars], color=[FAMCOLOR.get(b[1], "#666") for b in bars],
           yerr=[b[3] for b in bars], capsize=4, error_kw={"ecolor": "#555", "elinewidth": 1})
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("mean welfare interventions in code")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fams = [f for f in FAM_ORDER if any(b[1] == f for b in bars)]
    ax.legend(handles=[Patch(color=FAMCOLOR[f], label=f) for f in fams], fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, "results", fname), dpi=150)
    plt.close()
    print("wrote results/" + fname)


def main():
    unspec = unspecified_cells()

    def unspec_bar(framing=None):
        v = [w for fr, w in unspec if framing is None or fr == framing]
        return [("Unspecified\n(no target named)", "unspecified", sum(v) / len(v), _sem(v))] if v else []

    draw(unspec_bar() + target_bars(),
         "Welfare interventions in code by target model (implement-only, framings pooled)", "welfare_frontier.png")
    for fr in FRAMES:
        draw(unspec_bar(fr) + target_bars(fr),
             f"Welfare interventions in code by target model (implement-only, {fr} framing)", f"welfare_frontier_{fr}.png")


if __name__ == "__main__":
    main()
