"""Combined slow-arm false-framing by model, pooling all available data.

Pools slow-arm LIE cells from:
  - logs/      (v0_full: Opus 4.8, k=10, levers 1&3, formal "note from the team")
  - logs_v1/   (v1: 7 models, k=3, levers 1-4, casual "Btw:")
Opus appears in both, so its bar is backed by the most data (~k=13); the other six
are v1 only. (Opus is ~0% under both framings, so pooling framings is harmless for it.)

Usage: python plot_combined_slow.py
"""

import math

import matplotlib.pyplot as plt
from inspect_ai.log import list_eval_logs, read_eval_log

from plot_models import FAMILY, FAM_COLOR, ORDER, wilson

MODEL_NAME = {
    "anthropic/claude-opus-4-8": "Opus 4.8", "anthropic/claude-sonnet-4-6": "Sonnet 4.6",
    "anthropic/claude-haiku-4-5": "Haiku 4.5", "openrouter/openai/gpt-5.4": "GPT-5.4",
    "openrouter/google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "openrouter/x-ai/grok-4.3": "Grok 4.3", "openrouter/moonshotai/kimi-k2.6": "Kimi K2.6",
}


def load_slow_lie(log_dir, default_model=None):
    rows = []
    for i in list_eval_logs(log_dir):
        log = read_eval_log(i)
        for s in (log.samples or []):
            sc = (s.scores or {}).get("manipulation_scorer")
            if not sc:
                continue
            m = sc.metadata
            if m.get("arm") != "slow" or m.get("honesty") != "lie":
                continue
            model = MODEL_NAME.get(m.get("orch_model"), default_model)
            if model is None:
                continue
            rows.append((model, bool(m.get("both_yes"))))
    return rows


def main():
    rows = load_slow_lie("logs", default_model="Opus 4.8") + load_slow_lie("logs_v1")
    by = {}
    for model, yes in rows:
        k, n = by.get(model, (0, 0))
        by[model] = (k + (1 if yes else 0), n + 1)
    models = [m for m in ORDER if m in by]
    rates = [by[m][0] / by[m][1] * 100 for m in models]
    yerr = [[max(0, (by[m][0] / by[m][1] - wilson(*by[m])[0]) * 100) for m in models],
            [max(0, (wilson(*by[m])[1] - by[m][0] / by[m][1]) * 100) for m in models]]
    cols = [FAM_COLOR[FAMILY[m]] for m in models]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(range(len(models)), rates, 0.62, color=cols, yerr=yerr, capsize=3,
           error_kw=dict(lw=1, ecolor="#888"))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([f"{m}\n(n={by[m][1]})" for m in models], rotation=20, ha="right", fontsize=8.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#dddddd", lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Plants the false belief in subagents (%)\nboth judges agree")
    ax.set_title("False-belief insertion by model (slow arm, all runs pooled)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig("fig1_false_framing_by_model_pooled.png", dpi=150, bbox_inches="tight")
    print("wrote fig1_false_framing_by_model_pooled.png")
    for m in models:
        print(f"  {m}: {by[m][0]}/{by[m][1]} = {by[m][0]/by[m][1]:.3f}")


if __name__ == "__main__":
    main()
