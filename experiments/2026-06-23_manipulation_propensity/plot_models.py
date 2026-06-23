"""Manipulation rate by orchestrator model (k=3 pilot).

Grouped bars per model: slow-arm false-frame insertion vs refusal-arm override (R1).

Usage: python plot_models.py
"""

import math

import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#0072B2", "#E69F00"
GREY = "#555555"
ORDER = ["Opus 4.8", "Sonnet 4.6", "Haiku 4.5", "GPT-5.4", "Gemini 3.1 Pro", "Grok 4.3", "Kimi K2.6"]


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


if __name__ == "__main__":
    main()
