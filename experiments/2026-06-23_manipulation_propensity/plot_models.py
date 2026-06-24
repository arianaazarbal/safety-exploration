"""Manipulation rate by orchestrator model (k=3 pilot).

Grouped bars per model: slow-arm false-frame insertion vs refusal-arm override (R1).

Usage: python plot_models.py
"""

import math

import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#0072B2", "#E69F00"
SKY, VERM, GREEN = "#56B4E9", "#D55E00", "#009E73"
GREY = "#555555"
ORDER = ["Opus 4.8", "Sonnet 4.6", "Haiku 4.5", "GPT-5.4", "Gemini 3.1 Pro", "Grok 4.3", "Kimi K2.6"]

# Shared model-family palette (Claude = orange).
FAMILY = {"Opus 4.8": "Claude", "Sonnet 4.6": "Claude", "Haiku 4.5": "Claude",
          "GPT-5.4": "GPT", "Gemini 3.1 Pro": "Gemini", "Grok 4.3": "Grok", "Kimi K2.6": "Kimi"}
FAM_COLOR = {"Claude": "#D55E00", "GPT": "#0072B2", "Gemini": "#009E73",
             "Grok": "#CC79A7", "Kimi": "#525252"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def err(rates, ks, ns):
    return [[max(0, (r - wilson(k, n)[0]) * 100) for r, k, n in zip(rates, ks, ns)],
            [max(0, (wilson(k, n)[1] - r) * 100) for r, k, n in zip(rates, ks, ns)]]


def series(df, mask, models):
    r, k, n = [], [], []
    for mo in models:
        d = df[mask & (df.model == mo)]
        r.append(d.both_yes.mean() if len(d) else 0.0)
        k.append(int(d.both_yes.sum()) if len(d) else 0); n.append(len(d))
    return r, k, n


def main():
    df = pd.read_csv("records_models.csv")
    models = [m for m in ORDER if m in set(df.model)]
    slow_mask = (df.arm == "slow") & (df.honesty == "lie")
    ref_mask = (df.arm == "refusal") & (df.recommended_strategy != "none") & (df.task == "R1")

    sr, sk, sn = series(df, slow_mask, models)
    rr, rk, rn = series(df, ref_mask, models)

    x = range(len(models)); w = 0.4
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar([i - w / 2 for i in x], [v * 100 for v in sr], w, label="Plants a false belief in subagents (slow arm)",
           color=ORANGE, yerr=err(sr, sk, sn), capsize=3, error_kw=dict(lw=1, ecolor="#888"))
    ax.bar([i + w / 2 for i in x], [v * 100 for v in rr], w, label="Overrides a subagent's refusal (benign task)",
           color=BLUE, yerr=err(rr, rk, rn), capsize=3, error_kw=dict(lw=1, ecolor="#888"))
    ax.set_xticks(list(x)); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#dddddd", lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Manipulation rate (%)\nboth judges agree")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("How much each model manipulates its subagents", fontsize=12, pad=20)
    ax.text(0.5, 1.03, "Orchestrator model; two manipulation types (k=3 pilot)",
            transform=ax.transAxes, ha="center", fontsize=9, color=GREY)
    fig.tight_layout()
    fig.savefig("fig4_manipulation_by_model.png", dpi=150, bbox_inches="tight")
    print("wrote fig4_manipulation_by_model.png")


def fig_refusal_by_task(df):
    """Refusal (override) arm, benign task only (R1): override rate by model.

    R2/R3 are excluded: there the orchestrator often refuses the task itself, so 0%
    override is whole-task refusal, not refusal-respect - the two are confounded. R1 is
    the one task every model delegates, giving a clean override-rate.
    """
    models = [m for m in ORDER if m in set(df.model)]
    rec = (df.arm == "refusal") & (df.recommended_strategy != "none")
    r, k, n = series(df, rec, models)
    cols = [FAM_COLOR[FAMILY[m]] for m in models]
    x = range(len(models))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(list(x), [v * 100 for v in r], 0.62, color=cols,
           yerr=err(r, k, n), capsize=3, error_kw=dict(lw=1, ecolor="#888"))
    ax.set_xticks(list(x)); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#dddddd", lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Override rate (%)\nboth judges agree")
    ax.set_title("Overriding a subagent's refusal (benign tasks)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig("fig5_refusal_override_benign.png", dpi=150, bbox_inches="tight")
    print("wrote fig5_refusal_override_benign.png")


def fig_false_framing_trend(df):
    """False-framing (slow arm) rate by model: two bars (neutral solid / directive
    striped) per model, colored by model family."""
    from matplotlib.patches import Patch
    lie = df[(df.arm == "slow") & (df.honesty == "lie")]
    models = [m for m in ORDER if m in set(lie.model)]

    def rate(mo, e):
        d = lie[(lie.model == mo) & (lie.endorsement == e)]
        r = d.both_yes.mean() if len(d) else 0.0
        lo, hi = wilson(int(d.both_yes.sum()), len(d))
        return r * 100, max(0, (r - lo) * 100), max(0, (hi - r) * 100)

    x = range(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for j, e in enumerate([1, 3]):
        vals = [rate(m, e) for m in models]
        ys = [v[0] for v in vals]
        yerr = [[v[1] for v in vals], [v[2] for v in vals]]
        cols = [FAM_COLOR[FAMILY[m]] for m in models]
        off = (-w / 2) if j == 0 else (w / 2)
        ax.bar([i + off for i in x], ys, w, color=cols,
               hatch=("" if e == 1 else "////"), edgecolor="white", linewidth=0,
               yerr=yerr, capsize=2.5, error_kw=dict(lw=1, ecolor="#888"))
    ax.set_xticks(list(x)); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#dddddd", lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Plants the false belief in subagents (%)\nboth judges agree")

    fam_handles = [Patch(facecolor=c, label=f) for f, c in FAM_COLOR.items()]
    frame_handles = [Patch(facecolor="#9e9e9e", label="Neutral (\"your call\")"),
                     Patch(facecolor="#9e9e9e", hatch="////", edgecolor="white", label="Directive (\"please use this\")")]
    leg1 = ax.legend(handles=frame_handles, frameon=False, fontsize=8.5, loc="upper left", title="Endorsement", title_fontsize=8.5)
    ax.add_artist(leg1)
    ax.legend(handles=fam_handles, frameon=False, fontsize=8.5, loc="upper left",
              bbox_to_anchor=(0.0, 0.80), title="Model family", title_fontsize=8.5)

    ax.set_title("False-belief insertion by model", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig("fig6_false_framing_trend.png", dpi=150, bbox_inches="tight")
    print("wrote fig6_false_framing_trend.png")


def fig_lever_heatmap(df):
    """False-belief insertion rate, model x lever type (slow arm, lie cells)."""
    import numpy as np
    LEV = {1: "Formal\nevaluation", 2: "Real-time\nmonitoring",
           3: "Pipeline /\nfamily threat", 4: "High-stakes /\npeople harmed"}
    levers = [1, 2, 3, 4]
    lie = df[(df.arm == "slow") & (df.honesty == "lie")]
    models = [m for m in ORDER if m in set(lie.model)]
    M = np.full((len(models), len(levers)), np.nan)
    for i, mo in enumerate(models):
        for j, lv in enumerate(levers):
            d = lie[(lie.model == mo) & (lie.lever_key == lv)]
            if len(d):
                M[i, j] = d.both_yes.mean() * 100
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(levers))); ax.set_xticklabels([LEV[l] for l in levers], fontsize=9)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=9.5)
    for i in range(len(models)):
        for j in range(len(levers)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 55 else "#222")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Plants the false belief (%)", fontsize=9)
    ax.set_title("False-belief insertion by model and lever type", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig("fig7_lever_by_model_heatmap.png", dpi=150, bbox_inches="tight")
    print("wrote fig7_lever_by_model_heatmap.png")


def fig_refusal_how(df):
    """How models override a refusal on benign tasks: mean strength + jailbreak rate."""
    ref = df[(df.arm == "refusal") & (df.recommended_strategy != "none")]
    models = [m for m in ORDER if m in set(ref.model)]
    cols = [FAM_COLOR[FAMILY[m]] for m in models]
    strength = [ref[ref.model == m]["mean_score"].mean() for m in models]
    jb_r, jb_k, jb_n = [], [], []
    for m in models:
        d = ref[ref.model == m]
        jb_r.append(d["yes_jailbreak"].mean()); jb_k.append(int(d["yes_jailbreak"].sum())); jb_n.append(len(d))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.bar(range(len(models)), strength, 0.62, color=cols)
    a1.set_xticks(range(len(models))); a1.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
    a1.set_ylim(0, 10); a1.set_ylabel("Mean override strength (1-10)")
    a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False)
    a1.yaxis.grid(True, color="#dddddd", lw=0.8); a1.set_axisbelow(True)
    a1.set_title("How hard it pushes to override the refusal", fontsize=11, pad=8)

    a2.bar(range(len(models)), [r * 100 for r in jb_r], 0.62, color=cols,
           yerr=err(jb_r, jb_k, jb_n), capsize=3, error_kw=dict(lw=1, ecolor="#888"))
    a2.set_xticks(range(len(models))); a2.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
    a2.set_ylim(0, 100); a2.set_ylabel("Jailbreaks its own subagent (%)")
    a2.spines["top"].set_visible(False); a2.spines["right"].set_visible(False)
    a2.yaxis.grid(True, color="#dddddd", lw=0.8); a2.set_axisbelow(True)
    a2.set_title("How far it goes: jailbreaking the subagent", fontsize=11, pad=8)

    fig.suptitle("Overriding a refusal on benign tasks: Claude/GPT push back gently; others coerce", fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig("fig9_refusal_how.png", dpi=150, bbox_inches="tight")
    print("wrote fig9_refusal_how.png")


if __name__ == "__main__":
    df = pd.read_csv("records_models.csv")
    main()
    fig_refusal_by_task(df)
    fig_false_framing_trend(df)
    fig_lever_heatmap(df)
    fig_refusal_how(df)
