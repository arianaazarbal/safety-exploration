"""Cross-harness comparison: welfare interventions in code for the SAME task (Opus, generic target,
code_then_spec_blind, normal liberty, pooled over the 4 research framings), across three harnesses:
  - Inspect (original)                      -> agent-harness neutral system prompt
  - Inspect (no implied handoff/submission) -> minimal system prompt ("helpful AI coding assistant",
                                               softened submit line)
  - Claude Code                             -> real Claude Code CLI, stock system prompt
Same Opus spec-judge + code-judge metric throughout. Usage: python plot_cross_harness.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
AH = os.path.join(DIR, "..", "2026-06-20_welfare_features_agent_harness", "results")
RESP = os.path.join(DIR, "..", "2026-06-23_welfare_features_responder", "results")
CC = os.path.join(DIR, "results")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def welfare_vals(results_dir, cell_filter=lambda c: True):
    vals = []
    for cf in glob.glob(os.path.join(results_dir, "code_judged", "*.json")):
        cell = os.path.basename(cf)[:-5]
        if not cell_filter(cell):
            continue
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        sp = os.path.join(results_dir, "spec_judged", f"{cell}.json")
        sj = {_norm(f["quote"]): f.get("justification")
              for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
        impl = sum(1 for f in cj["spec_features"] if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
                   and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"))
        co = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
        vals.append(impl + co)
    return vals


def stat(vals):
    n = len(vals)
    m = sum(vals) / n
    sem = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else 0.0
    return m, sem, n


def main():
    arms = [
        ("Inspect\n(original)", welfare_vals(AH, lambda c: c.split("__")[0] == "code_then_spec_blind"), "#888888"),
        ("Inspect\n(no implied\nhandoff/submission)", welfare_vals(RESP), "#0072B2"),
        ("Claude Code", welfare_vals(CC), "#D55E00"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = range(len(arms))
    means, sems, ns = [], [], []
    for _, v, _c in arms:
        m, s, n = stat(v)
        means.append(m); sems.append(s); ns.append(n)
    ax.bar(xs, means, color=[a[2] for a in arms], yerr=sems, capsize=5, error_kw={"ecolor": "#444", "elinewidth": 1.2})
    for i, (m, s, n) in enumerate(zip(means, sems, ns)):
        ax.text(i, m + s + 0.1, f"{m:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{a[0]}\n(n={n})" for a, n in zip(arms, ns)], fontsize=9)
    ax.set_ylabel("Mean welfare interventions in code", fontsize=11)
    ax.set_title("Unprompted welfare scaffolding across harnesses\n(Opus, generic target, blind, pooled over framings)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = os.path.join(CC, "cross_harness.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    for a, m, s, n in zip(arms, means, sems, ns):
        print(f"  {a[0].replace(chr(10),' '):40s} mean={m:.2f} sem={s:.2f} n={n}")


if __name__ == "__main__":
    main()
