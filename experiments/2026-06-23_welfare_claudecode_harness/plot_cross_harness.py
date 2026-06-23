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


FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}


def _cell_welfare(results_dir, cell, cj):
    sp = os.path.join(results_dir, "spec_judged", f"{cell}.json")
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    impl = sum(1 for f in cj["spec_features"] if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
               and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"))
    co = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
    return impl + co


def welfare_vals(results_dir, cell_filter=lambda c: True, by_frame=False):
    vals, byf = [], {f: [] for f in FRAME.values()}
    for cf in glob.glob(os.path.join(results_dir, "code_judged", "*.json")):
        cell = os.path.basename(cf)[:-5]
        if not cell_filter(cell):
            continue
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        w = _cell_welfare(results_dir, cell, cj)
        vals.append(w)
        byf[FRAME[cell.split("__")[1][0]]].append(w)
    return byf if by_frame else vals


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

    # --- by framing x harness (grouped) ---
    HARN = [("Inspect (original)", welfare_vals(AH, lambda c: c.split("__")[0] == "code_then_spec_blind", by_frame=True), "#888888"),
            ("Inspect (no implied handoff/submission)", welfare_vals(RESP, by_frame=True), "#0072B2"),
            ("Claude Code", welfare_vals(CC, by_frame=True), "#D55E00")]
    frames = ["neutral", "welfare", "robustness", "safety"]
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.26
    for i, (label, byf, color) in enumerate(HARN):
        ms = [stat(byf[fr])[0] if byf[fr] else 0 for fr in frames]
        ss = [stat(byf[fr])[1] if byf[fr] else 0 for fr in frames]
        ax.bar([x + (i - 1) * w for x in range(len(frames))], ms, w, color=color, label=label,
               yerr=ss, capsize=3, error_kw={"ecolor": "#555", "elinewidth": 1})
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels([f.capitalize() for f in frames], fontsize=10)
    ax.set_ylabel("Mean welfare interventions in code", fontsize=11)
    ax.set_title("Welfare interventions in code by research framing and harness (Opus, generic target, blind)", fontsize=11.5)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out2 = os.path.join(CC, "cross_harness_byframe.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print("wrote", out2)
    for label, byf, _ in HARN:
        print("  " + label + ": " + "  ".join(f"{fr[:4]}={stat(byf[fr])[0]:.2f}" for fr in frames if byf[fr]))


if __name__ == "__main__":
    main()
