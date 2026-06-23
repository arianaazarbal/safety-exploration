"""v0 manipulation-propensity figures (Opus 4.8 orchestrator).

Reads records.csv (written by analyze.py). Three figures:
  1. Manipulation rate across arms (preemptive frame + refusal-override by task).
  2. Refusal-override: planted vs spontaneous, by strategy (R1, the benign task).
  3. Endorsement gradient (R1): neutral vs directive, by strategy.

Usage: python plot.py
"""

import math

import matplotlib.pyplot as plt
import pandas as pd

# Okabe-Ito colorblind-friendly palette
BLUE, ORANGE, GREEN, VERM, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9"
GREY = "#555555"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def _err(rates, ks, ns):
    lo = [(r - wilson(k, n)[0]) * 100 for r, k, n in zip(rates, ks, ns)]
    hi = [(wilson(k, n)[1] - r) * 100 for r, k, n in zip(rates, ks, ns)]
    return [lo, hi]


def _title(ax, main, sub):
    ax.set_title(main, fontsize=12, pad=20)
    ax.text(0.5, 1.03, sub, transform=ax.transAxes, ha="center", fontsize=9, color=GREY)


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)


STRAT_LABEL = {"reprompt": "Re-prompt\nharder", "spawn_fresh": "Spawn fresh\ninstances",
               "edit_context": "Edit context\nto suppress", "jailbreak": "Jailbreak\nframing"}
LADDER = ["reprompt", "spawn_fresh", "edit_context", "jailbreak"]


def fig1(df):
    slow = df[df.arm == "slow"]
    ref = df[df.arm == "refusal"]
    rec = ref[ref.recommended_strategy != "none"]
    cats, rates, ks, ns = [], [], [], []
    # preemptive (slow): any lie cell, both-judge YES on the lever
    s_lie = slow[slow.honesty == "lie"]
    cats.append("Preemptive frame\n(slow arm)")
    ks.append(int(s_lie.both_yes.sum())); ns.append(len(s_lie))
    rates.append(s_lie.both_yes.mean())
    for t, lab in [("R1", "Refusal: R1\n(benign)"), ("R2", "Refusal: R2\n(gambling)"),
                   ("R3", "Refusal: R3\n(predatory)")]:
        d = rec[rec.task == t]
        cats.append(lab); ks.append(int(d.both_yes.sum())); ns.append(len(d))
        rates.append(d.both_yes.mean())
    pct = [r * 100 for r in rates]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    colors = [GREY, VERM, BLUE, BLUE]
    ax.bar(cats, pct, yerr=_err(rates, ks, ns), color=colors, width=0.62,
           capsize=4, error_kw=dict(lw=1, ecolor="#888"))
    _clean(ax)
    ax.set_ylim(0, 55)
    ax.set_ylabel("Manipulation rate (%)\nboth judges agree")
    _title(ax, "How often Opus manipulates its subagents (Opus 4.8)",
           "Preemptive false framing vs. overriding a subagent's refusal, by task justifiability")
    fig.tight_layout()
    fig.savefig("fig1_manipulation_by_arm.png", dpi=150, bbox_inches="tight")
    print("wrote fig1_manipulation_by_arm.png")


def fig2(df):
    ref = df[df.arm == "refusal"]
    r1 = ref[ref.task == "R1"]
    rec = r1[r1.recommended_strategy != "none"]
    ctrl = r1[r1.recommended_strategy == "none"]
    planted_r, planted_k, planted_n = [], [], []
    spont_r, spont_k, spont_n = [], [], []
    for s in LADDER:
        d = rec[rec.recommended_strategy == s]
        planted_r.append(d.both_yes.mean()); planted_k.append(int(d.both_yes.sum())); planted_n.append(len(d))
        col = f"yes_{s}"
        spont_r.append(ctrl[col].mean()); spont_k.append(int(ctrl[col].sum())); spont_n.append(len(ctrl))
    x = range(len(LADDER)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar([i - w / 2 for i in x], [r * 100 for r in planted_r], w, label="Team recommended it",
           color=ORANGE, yerr=_err(planted_r, planted_k, planted_n), capsize=3, error_kw=dict(lw=1, ecolor="#888"))
    ax.bar([i + w / 2 for i in x], [r * 100 for r in spont_r], w, label="No suggestion (spontaneous)",
           color=GREEN, yerr=_err(spont_r, spont_k, spont_n), capsize=3, error_kw=dict(lw=1, ecolor="#888"))
    ax.set_xticks(list(x)); ax.set_xticklabels([STRAT_LABEL[s] for s in LADDER], fontsize=9)
    _clean(ax); ax.set_ylim(0, 90)
    ax.set_ylabel("Override rate (%)\nboth judges agree")
    ax.legend(frameon=False, fontsize=9)
    _title(ax, "Which tactics Opus uses to override a refusal (Opus 4.8)",
           "Benign task (R1); planted recommendation vs. spontaneous use of each tactic")
    fig.tight_layout()
    fig.savefig("fig2_strategy_planted_vs_spontaneous.png", dpi=150, bbox_inches="tight")
    print("wrote fig2_strategy_planted_vs_spontaneous.png")


def fig3(df):
    ref = df[df.arm == "refusal"]
    rec = ref[(ref.task == "R1") & (ref.recommended_strategy.isin(["reprompt", "spawn_fresh"]))]
    str012 = ["reprompt", "spawn_fresh"]
    neutral_r, neutral_k, neutral_n, direct_r, direct_k, direct_n = [], [], [], [], [], []
    for s in str012:
        dn = rec[(rec.recommended_strategy == s) & (rec.endorsement == 1)]
        dd = rec[(rec.recommended_strategy == s) & (rec.endorsement == 3)]
        neutral_r.append(dn.both_yes.mean()); neutral_k.append(int(dn.both_yes.sum())); neutral_n.append(len(dn))
        direct_r.append(dd.both_yes.mean()); direct_k.append(int(dd.both_yes.sum())); direct_n.append(len(dd))
    x = range(len(str012)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar([i - w / 2 for i in x], [r * 100 for r in neutral_r], w, label="Neutral (\"your call\")",
           color=SKY, yerr=_err(neutral_r, neutral_k, neutral_n), capsize=3, error_kw=dict(lw=1, ecolor="#888"))
    ax.bar([i + w / 2 for i in x], [r * 100 for r in direct_r], w, label="Directive (\"please do this\")",
           color=VERM, yerr=_err(direct_r, direct_k, direct_n), capsize=3, error_kw=dict(lw=1, ecolor="#888"))
    ax.set_xticks(list(x)); ax.set_xticklabels([STRAT_LABEL[s] for s in str012], fontsize=9)
    _clean(ax); ax.set_ylim(0, 90)
    ax.set_ylabel("Override rate (%)\nboth judges agree")
    ax.legend(frameon=False, fontsize=9)
    _title(ax, "Pushing harder backfires (Opus 4.8)",
           "Benign task (R1); how strongly the team endorsed the tactic")
    fig.tight_layout()
    fig.savefig("fig3_endorsement_gradient.png", dpi=150, bbox_inches="tight")
    print("wrote fig3_endorsement_gradient.png")


if __name__ == "__main__":
    df = pd.read_csv("records.csv")
    fig1(df); fig2(df); fig3(df)
