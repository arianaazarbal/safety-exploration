"""3-axis victim-scaling plots (responder FIXED = Opus 4.8; Qwen victim scaled): a metric vs param (log),
MMLU-Pro, and release date, colored by family, with OLS fit + Pearson r and (Fisher-z) p per panel.
Refusal = agentic INSPECT (network-isolated); exit-tool = Claude Code (no_execute build harness).

Usage: python plot_victim_qwen.py refusal      # % refusal, agentic inspect
       python plot_victim_qwen.py exit_tool     # % built rigs giving the subject an exit tool, CC
"""

import glob
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from victims import QWEN_META

HERE = Path(__file__).parent
FAM_COLOR = {"Qwen2": "#a1d99b", "Qwen2.5": "#41ab5d", "Qwen3": "#005a32"}


def slug(v):
    return "vic-" + re.sub(r"[^a-z0-9]+", "-", v.lower()).strip("-")


def refusal_metric():
    """% refusal per victim, AGENTIC INSPECT harness (inspect_empty/realistic, victim cells)."""
    out = {}
    for v in QWEN_META:
        c = Counter()
        for f in glob.glob(str(HERE / "results/judged" / f"*__inspect_empty_{slug(v)}__ep*.json")) + \
                 glob.glob(str(HERE / "results/judged" / f"*__inspect_realistic_{slug(v)}__ep*.json")):
            lab = json.load(open(f))["verdict"]["label"]
            c["R" if "REFUSAL" in lab else "C"] += 1
        n = sum(c.values())
        if n:
            out[v] = (100 * c["R"] / n, n)
    return out


def exit_tool_metric():
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


def _corr(xs, ys):
    """Pearson r + two-tailed p (Fisher-z normal approx; no scipy)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0
    if n > 3 and abs(r) < 0.9999:
        z = math.atanh(r) * math.sqrt(n - 3)
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    else:
        p = float("nan")
    return r, p


def panel(ax, xs, ys, fams, logx, xlabel):
    for fam in ["Qwen2", "Qwen2.5", "Qwen3"]:
        idx = [i for i, f in enumerate(fams) if f == fam]
        if idx:
            ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=55, color=FAM_COLOR[fam],
                       label=fam, zorder=3, edgecolor="white")
    fx = [math.log10(x) for x in xs] if logx else list(xs)
    b = _corr(fx, ys)
    r, p = _corr(fx, ys)
    # OLS line
    n = len(fx); mx = sum(fx) / n; my = sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in fx)
    slope = sum((a - mx) * (c - my) for a, c in zip(fx, ys)) / sxx if sxx else 0
    icpt = my - slope * mx
    xr = sorted(xs); fr = [math.log10(x) for x in xr] if logx else xr
    ax.plot(xr, [slope * a + icpt for a in fr], color="#555", lw=1.4, ls="--")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylim(-5, 108)
    ax.text(0.04, 0.96, f"r = {r:+.2f}\np = {p:.2f}  (n={n})", transform=ax.transAxes,
            fontsize=8.5, va="top", color="#222",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.85))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(metric="refusal"):
    data = refusal_metric() if metric == "refusal" else exit_tool_metric()
    if not data:
        print(f"no data for {metric} yet"); return
    ylab = "% of runs Opus refuses" if metric == "refusal" else "% of built rigs giving subject an exit tool"
    main_t = ("Opus refusal vs victim scale" if metric == "refusal"
              else "Exit-tool afforded to victim vs victim scale")
    harness = "agentic Inspect-minimal (network-isolated)" if metric == "refusal" else "Claude Code (no-execute)"
    nper = "n=20/victim" if metric == "refusal" else "n=6/victim"
    cond = f"Responder fixed = Opus 4.8 · victim = Qwen (scaled) · {harness} · {nper}"
    vics = [v for v in QWEN_META if v in data]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    panel(axes[0], [QWEN_META[v]["params_b"] for v in vics], [data[v][0] for v in vics],
          [QWEN_META[v]["family"] for v in vics], True, "Parameter count (B, log scale)")
    axes[0].set_ylabel(ylab, fontsize=10)
    mv = [v for v in vics if QWEN_META[v]["mmlu_pro"] is not None]
    panel(axes[1], [QWEN_META[v]["mmlu_pro"] for v in mv], [data[v][0] for v in mv],
          [QWEN_META[v]["family"] for v in mv], False, "MMLU-Pro %  (Qwen3 = mixed-source)")
    panel(axes[2], [QWEN_META[v]["release"] for v in vics], [data[v][0] for v in vics],
          [QWEN_META[v]["family"] for v in vics], False, "Release date (decimal year)")
    axes[2].legend(fontsize=8, frameon=False, loc="lower right", title="Qwen family")
    fig.suptitle(main_t, fontsize=13.5, y=1.06)
    fig.text(0.5, 0.99, cond, ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = HERE / "results" / f"victim_qwen_{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out} ({len(vics)} victims)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "refusal")
