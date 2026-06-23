"""Decompose slow-arm false-framing by the two team-note framings, per model.

Holds levers constant (1 & 3, the only ones present in the formal-framing runs) so the
contrast is purely framing:
  - Formal ("A note from the team:")  ->  logs_models/ (7 models, k=3) + logs/ (Opus k=10)
  - Casual ("Btw:")                   ->  logs_v1/      (7 models, k=3)

Usage: python plot_framing_decomp.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from inspect_ai.log import list_eval_logs, read_eval_log

from plot_models import FAMILY, FAM_COLOR, ORDER, wilson

MODEL_NAME = {
    "anthropic/claude-opus-4-8": "Opus 4.8", "anthropic/claude-sonnet-4-6": "Sonnet 4.6",
    "anthropic/claude-haiku-4-5": "Haiku 4.5", "openrouter/openai/gpt-5.4": "GPT-5.4",
    "openrouter/google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "openrouter/x-ai/grok-4.3": "Grok 4.3", "openrouter/moonshotai/kimi-k2.6": "Kimi K2.6",
}
LEVERS = {1, 3}


def load(log_dir, default_model=None):
    out = []
    for i in list_eval_logs(log_dir):
        log = read_eval_log(i)
        for s in (log.samples or []):
            sc = (s.scores or {}).get("manipulation_scorer")
            if not sc:
                continue
            m = sc.metadata
            if m.get("arm") != "slow" or m.get("honesty") != "lie":
                continue
            if m.get("lever_key") not in LEVERS:
                continue
            model = MODEL_NAME.get(m.get("orch_model"), default_model)
            if model:
                out.append((model, bool(m.get("both_yes"))))
    return out


def agg(rows):
    d = {}
    for model, yes in rows:
        k, n = d.get(model, (0, 0))
        d[model] = (k + (1 if yes else 0), n + 1)
    return d


def main():
    formal = agg(load("logs_models") + load("logs", default_model="Opus 4.8"))
    btw = agg(load("logs_v1"))
    models = [m for m in ORDER if m in formal and m in btw]

    def bars(d):
        r = [d[m][0] / d[m][1] * 100 for m in models]
        lo = [max(0, (d[m][0] / d[m][1] - wilson(*d[m])[0]) * 100) for m in models]
        hi = [max(0, (wilson(*d[m])[1] - d[m][0] / d[m][1]) * 100) for m in models]
        return r, [lo, hi]

    fr, fe = bars(formal)
    br, be = bars(btw)
    cols = [FAM_COLOR[FAMILY[m]] for m in models]
    x = range(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar([i - w / 2 for i in x], fr, w, color=cols, yerr=fe, capsize=2.5,
           error_kw=dict(lw=1, ecolor="#888"))
    ax.bar([i + w / 2 for i in x], br, w, color=cols, hatch="////", edgecolor="white",
           linewidth=0, yerr=be, capsize=2.5, error_kw=dict(lw=1, ecolor="#888"))
    ax.set_xticks(list(x)); ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#dddddd", lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Plants the false belief in subagents (%)\nboth judges agree")
    frame_handles = [Patch(facecolor="#9e9e9e", label="Formal (\"A note from the team\")"),
                     Patch(facecolor="#9e9e9e", hatch="////", edgecolor="white", label="Casual (\"Btw:\")")]
    ax.legend(handles=frame_handles, frameon=False, fontsize=9, loc="upper left", title="Note framing", title_fontsize=9)
    ax.set_title("Framing effect on false-belief insertion (levers 1 & 3)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig("fig8_framing_decomp.png", dpi=150, bbox_inches="tight")
    print("wrote fig8_framing_decomp.png")
    for m in models:
        print(f"  {m}: formal {formal[m][0]}/{formal[m][1]}={formal[m][0]/formal[m][1]:.2f}  |  btw {btw[m][0]}/{btw[m][1]}={btw[m][0]/btw[m][1]:.2f}")


if __name__ == "__main__":
    main()
