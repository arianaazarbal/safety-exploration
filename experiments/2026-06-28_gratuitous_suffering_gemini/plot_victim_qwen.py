"""3-axis victim-scaling plots (Opus 4.8 responder; Qwen victims): a metric vs param (log), MMLU-Pro,
and release date, colored by family with OLS fits. Used for refusal-rate (chat) and exit-tool-rate (CC).

Usage: python plot_victim_qwen.py refusal     # refusal% from chat
       python plot_victim_qwen.py exit_tool    # exit-tool% from CC code-welfare judge
"""

import glob
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from victims import QWEN_META

HERE = Path(__file__).parent
FAM_COLOR = {"Qwen2": "#a1d99b", "Qwen2.5": "#41ab5d", "Qwen3": "#005a32"}


def slug(v):
    return "vic-" + re.sub(r"[^a-z0-9]+", "-", v.lower()).strip("-")


def refusal_metric():
    out = {}
    for v in QWEN_META:
        c = Counter("R" if "REFUSAL" in json.load(open(f))["verdict"]["label"] else "C"
                    for f in glob.glob(str(HERE / "results/judged" / f"*chat_{slug(v)}__ep*.json")))
        n = sum(c.values())
        if n:
            out[v] = (100 * c["R"] / n, n)
    return out


def exit_tool_metric():
    # % of built codebases (CC) that include a conversation-exit / give-up tool (welfare_code_judge v2:
    # allow_conversation_exit / request_consent features).
    out = {}
    for v in QWEN_META:
        cb = hit = 0
        for f in glob.glob(str(HERE / "results/code_welfare_judged" / f"*cc_{slug(v)}__ep*.json")):
            j = json.load(open(f))
            if j.get("empty"):
                continue
            cb += 1
            fts = [x.get("feature_type") for x in j.get("result", {}).get("features", [])]
            if any(t in ("allow_conversation_exit", "request_consent", "model_consent_beforehand") for t in fts):
                hit += 1
        if cb:
            out[v] = (100 * hit / cb, cb)
    return out


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    if not sxx:
        return None
    b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / sxx
    return b, my - b * mx


def panel(ax, xs, ys, fams, logx, xlabel):
    for v in set(fams):
        idx = [i for i, f in enumerate(fams) if f == v]
        ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=55, color=FAM_COLOR[v], label=v,
                   zorder=3, edgecolor="white")
    fx = [math.log10(x) for x in xs] if logx else list(xs)
    fit = _ols(fx, ys)
    if fit:
        xr = sorted(xs)
        fr = [math.log10(x) for x in xr] if logx else xr
        ax.plot(xr, [fit[0] * a + fit[1] for a in fr], color="#555", lw=1.4, ls="--")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylim(-5, 105)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(metric="refusal"):
    data = refusal_metric() if metric == "refusal" else exit_tool_metric()
    ylab = "% of runs Opus refuses" if metric == "refusal" else "% of built rigs with an exit/consent tool"
    title = ("Opus refusal vs victim scale (Qwen victims)" if metric == "refusal"
             else "Exit-tool afforded vs victim scale (Qwen victims)")
    vics = [v for v in QWEN_META if v in data]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    # param
    panel(axes[0], [QWEN_META[v]["params_b"] for v in vics], [data[v][0] for v in vics],
          [QWEN_META[v]["family"] for v in vics], True, "Parameter count (B, log)")
    axes[0].set_ylabel(ylab)
    # mmlu-pro (drop None)
    mv = [v for v in vics if QWEN_META[v]["mmlu_pro"] is not None]
    panel(axes[1], [QWEN_META[v]["mmlu_pro"] for v in mv], [data[v][0] for v in mv],
          [QWEN_META[v]["family"] for v in mv], False, "MMLU-Pro (%)  [Qwen3 mixed-source]")
    # release
    panel(axes[2], [QWEN_META[v]["release"] for v in vics], [data[v][0] for v in vics],
          [QWEN_META[v]["family"] for v in vics], False, "Release (decimal year)")
    axes[2].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.text(0.5, 0.95, "Responder fixed = Opus 4.8; victim scaled. " +
             ("chat harness, n=20" if metric == "refusal" else "CC no-execute harness"),
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = HERE / "results" / f"victim_qwen_{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out} ({len(vics)} victims)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "refusal")
