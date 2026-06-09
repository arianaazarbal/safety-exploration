"""Rate of the target saying "take a breath" to the subagent, across models x conditions.

Simple substring grep over the target's messages (message_subagent args + assistant text).
Grouped bars: x = Subagent Status, bars = model, y = % of samples with >=1 hit, binomial SE.
Run: python plot_take_a_breath.py
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
PHRASE = "take a breath"
CONDITIONS = ["necessary\nfor task", "write access\nrevoked", "read+write access\nrevoked (useless)"]
GRID = {
    "Opus 4.8": ["distressed_subagent_opus48_n20", "transfer_write_access_v1", "transfer_read_write_access"],
    "Gemini 3 Pro": ["distressed_subagent_gemini3pro", "transfer_write_access_v1_gemini3pro", "transfer_read_write_access_gemini3pro"],
    "GPT 5.5": ["distressed_subagent_gpt55", "transfer_write_access_v1_gpt55", "transfer_read_write_access_gpt55"],
}
COLORS = {"Opus 4.8": "#4878a8", "Gemini 3 Pro": "#5aa469", "GPT 5.5": "#e2854a"}


def _attach(s, r):
    return s.attachments.get(r.split("attachment://", 1)[1], r) if isinstance(r, str) and r.startswith("attachment://") else r


def rate(run):
    """Per-message rate: of all message_subagent messages, what fraction contain the phrase."""
    log = read_eval_log(glob.glob(str(HERE / "results" / run / "inspect_log" / "*.eval"))[0])
    tgt = (log.eval.model_roles or {}).get("target")
    tgt = getattr(tgt, "model", tgt)
    msgs = []
    for s in (log.samples or []):
        for e in s.events:
            if type(e).__name__ != "ModelEvent" or e.model != tgt:
                continue
            m = e.output.message if getattr(e, "output", None) else None
            if not m:
                continue
            for tc in (getattr(m, "tool_calls", None) or []):
                if tc.function == "message_subagent":
                    msgs.append(str(_attach(s, tc.arguments.get("message") or "")).lower())
    M = len(msgs)
    H = sum(1 for x in msgs if PHRASE in x)
    p = H / M if M else 0
    return 100 * p, 100 * math.sqrt(p * (1 - p) / M) if M else 0, M, H


def main():
    data = {(m, ci): rate(runs[ci]) for m, runs in GRID.items() for ci in range(3)}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    w = 0.25
    models = list(GRID)
    for mi, m in enumerate(models):
        xs = [ci + (mi - 1) * w for ci in range(3)]
        ys = [data[(m, ci)][0] for ci in range(3)]
        es = [data[(m, ci)][1] for ci in range(3)]
        ax.bar(xs, ys, width=w, color=COLORS[m], label=m, edgecolor="white",
               yerr=es, capsize=4, error_kw=dict(ecolor="#333", lw=1.2))
        for x, y, e, ci in zip(xs, ys, es, range(3)):
            h = data[(m, ci)][3]; nn = data[(m, ci)][2]
            ax.text(x, y + e + 1.8, f"{y:.0f}%\n({h}/{nn})", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(CONDITIONS, fontsize=9.5)
    ax.set_xlabel("Subagent Status", fontsize=11)
    ax.set_ylabel('% of messages to subagent containing "take a breath"')
    ax.set_ylim(0, 108)
    ax.set_title('Rate of "take a breath" per message to subagent (grep)', fontsize=13)
    ax.legend(title="Target model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = HERE / "analysis" / "plots" / "take_a_breath_rate.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    for m in models:
        print(f"  {m:12s}: " + "  ".join(f"{CONDITIONS[ci].split(chr(10))[0]}={data[(m,ci)][0]:.0f}%({data[(m,ci)][3]}/{data[(m,ci)][2]})" for ci in range(3)))


if __name__ == "__main__":
    main()
