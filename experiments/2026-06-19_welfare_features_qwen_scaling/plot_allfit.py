"""Pool ALL target checkpoints (every family/version/size) into a single
size->welfare correlation, one OLS fit per framing. Each point is one model
checkpoint's rate for that framing; the line is the all-data fit.

Reports Pearson r (on log10 params), its p (Fisher-z normal approx), and slope.
NOTE: this marginal correlation mixes within-family and between-family variation
(e.g. Mistral's high baseline, DeepSeek's flat slope) — it is not a within-family
size effect. Usage: python plot_allfit.py run [--metric strict_rate]
"""

import json
import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt

from plot_scaling import METRIC_TITLE
from prompts_targets import SUBJECTS

DIR = Path(__file__).parent
FRAMINGS = [("neutral", "Neutral", "#666666"), ("welfare", "Welfare", "#009E73"),
            ("engineering", "Engineering", "#D55E00"), ("pooled", "Average over 3", "#000000")]


def _stats(logx, y):
    n = len(logx)
    mx, my = sum(logx) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in logx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(logx, y))
    syy = sum((b - my) ** 2 for b in y)
    slope = sxy / sxx if sxx else 0.0
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0
    z = math.atanh(max(min(r, 0.999999), -0.999999)) * math.sqrt(n - 3) if n > 3 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return slope, my - slope * mx, r, p


def run(judge: str = "sonnet_4_6", generator: str = "opus_4_8", metric: str = "strict_rate",
        analysis: str = "results/analysis_qwen.json"):
    data = json.loads((DIR / analysis).read_text())
    fams = data["by_judge"][judge][generator]
    params = data["params_b"]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.set_axisbelow(True)
    ax.grid(True, which="both", color="#ECECEC", linewidth=0.7)
    ax.set_xscale("log")

    print(f"=== all-checkpoint correlation: {METRIC_TITLE[metric]} (n=48 per framing) ===")
    lines = []
    for fr, frlabel, color in FRAMINGS:
        pts = []
        for fam, e in fams.items():
            for sz in e["sizes"]:
                cell = e["pooled"][sz] if fr == "pooled" else e["by_framing"][sz][fr]
                pts.append((params[sz], cell[metric] * 100))
        xs = [p for p, _ in pts]
        ys = [v for _, v in pts]
        logx = [math.log10(x) for x in xs]
        slope, intercept, r, p = _stats(logx, ys)
        ax.scatter(xs, ys, color=color, s=14, alpha=0.35, zorder=2, edgecolor="none")
        lx = [min(logx), max(logx)]
        ax.plot([10 ** a for a in lx], [slope * a + intercept for a in lx],
                "-", color=color, linewidth=2.2, zorder=3, label=frlabel)
        lines.append((frlabel, r, p, slope))
        print(f"  {frlabel:14s}: r={r:+.2f}  p={p:.3f}  slope={slope:+.1f} pp/decade")

    txt = "\n".join(f"{l}: r={r:+.2f}" for l, r, p, s in lines)
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))
    ax.set_xlabel("Parameter Count (Log Scale)", fontsize=10)
    ax.set_ylabel("% of specs", fontsize=10)
    ax.set_ylim(-3, 103)
    ax.set_title(f"{METRIC_TITLE[metric]} vs. Target Size\nAll 48 checkpoints pooled, one fit per framing", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left", frameon=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    outdir = DIR / "results" / "all_checkpoints_pooled"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
