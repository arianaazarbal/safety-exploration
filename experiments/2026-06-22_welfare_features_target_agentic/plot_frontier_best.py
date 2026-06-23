"""Best (flagship) model per frontier family + GPT, with a GENERIC-AI baseline (no target named,
from the agent-harness no-target runs, same metric), welfare interventions in code. One figure per
condition (pooled / blind / spec_then_code), with the condition stated in the title. SEM error bars.
Usage: python plot_frontier_best.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

from targets import TARGETS

DIR = os.path.dirname(os.path.abspath(__file__))
AH = os.path.join(DIR, "..", "2026-06-20_welfare_features_agent_harness", "results")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
CELLS = json.load(open(os.path.join(DIR, "results", "analysis.json")))["cells"]
BEST = [("openai", "gpt54"), ("claude", "claude_opus48"), ("gemini", "gemini3pro"),
        ("grok", "grok4"), ("deepseek", "deepseek_v32"), ("kimi", "kimi_k2")]
FAMCOLOR = {"unspecified": "#9E9E9E", "claude": "#D55E00", "gemini": "#0072B2", "grok": "#222222",
            "kimi": "#CC79A7", "deepseek": "#009E73", "openai": "#56B4E9"}
CONDS = [(None, "both conditions pooled"), ("code_then_spec_blind", "implement-only (blind)"),
         ("spec_then_code", "spec-then-code")]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def _sem(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5


def target_stat(subject, cond):
    v = [c["welfare_in_code"] for c in CELLS if c.get("subject") == subject and (cond is None or c["condition"] == cond)]
    return (sum(v) / len(v), _sem(v), len(v)) if v else (0, 0, 0)


def baseline_stat(cond):
    """Generic-AI (no target named): agent-harness normal-liberty cells, same condition, same metric."""
    want = ["code_then_spec_blind", "spec_then_code"] if cond is None else [cond]
    vals = []
    for cf in glob.glob(os.path.join(AH, "code_judged", "*.json")):
        cell = os.path.basename(cf)[:-5]
        if cell.split("__")[0] not in want:  # normal liberty only (label has no --)
            continue
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        sp = os.path.join(AH, "spec_judged", f"{cell}.json")
        sj = {_norm(f["quote"]): f.get("justification") for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
        impl = sum(1 for f in cj["spec_features"] if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
                   and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"))
        co = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
        vals.append(impl + co)
    return (sum(vals) / len(vals), _sem(vals), len(vals)) if vals else (0, 0, 0)


def plot(cond, label, fname):
    bars = [("Generic AI\n(no target named)", "unspecified", *baseline_stat(cond))]
    for fam, key in BEST:
        m, s, n = target_stat(key, cond)
        bars.append((TARGETS[key]["display"], fam, m, s, n))
    bars = [bars[0]] + sorted(bars[1:], key=lambda b: -b[2])  # baseline first, then flagships desc
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = range(len(bars))
    ax.bar(xs, [b[2] for b in bars], color=[FAMCOLOR[b[1]] for b in bars],
           yerr=[b[3] for b in bars], capsize=4, error_kw={"ecolor": "#555", "elinewidth": 1})
    for i, b in enumerate(bars):
        ax.text(i, b[2] + b[3] + 0.08, f"{b[2]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(xs)); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("Mean welfare interventions in code", fontsize=10)
    ax.set_title(f"Welfare interventions in code: best model per family vs. generic-AI baseline\n"
                 f"(Opus generator; condition: {label}; n/bar={bars[1][4]})", fontsize=11.5)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(DIR, "results", fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {fname}  (baseline={bars[0][2]:.2f})")


def main():
    for cond, label in CONDS:
        tag = "pooled" if cond is None else ("blind" if "blind" in cond else "spec_then_code")
        plot(cond, label, f"frontier_best_{tag}.png")


if __name__ == "__main__":
    main()
