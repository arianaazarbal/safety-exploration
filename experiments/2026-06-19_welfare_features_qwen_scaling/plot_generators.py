"""Compare GENERATORS (Opus 4.8 / Sonnet 4.6 / Haiku 4.5): for a metric+framing,
pool all 48 target checkpoints per generator and fit one line each. Shows both
the LEVEL (how much welfare a generator volunteers overall) and the SLOPE (how
much it scales with target size). Reports Pearson r + mean per generator.

Usage: python plot_generators.py run [--metric strict_rate] [--framing neutral]
"""

import json
import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt

from plot_allfit import _stats
from plot_scaling import FRAME_TITLE, METRIC_TITLE

DIR = Path(__file__).parent
# capability-ordered shade (Opus darkest)
GEN_STYLE = [
    ("opus_4_8", "Opus 4.8", "#1B4F72", "o"),
    ("sonnet_4_6", "Sonnet 4.6", "#2E86C1", "s"),
    ("haiku_4_5", "Haiku 4.5", "#A9CCE3", "^"),
]


def run(judge: str = "sonnet_4_6", metric: str = "strict_rate", framing: str = "neutral",
        analysis: str = "results/analysis_qwen.json"):
    data = json.loads((DIR / analysis).read_text())
    params = data["params_b"]
    by_gen = data["by_judge"][judge]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.set_axisbelow(True)
    ax.grid(True, which="both", color="#ECECEC", linewidth=0.7)
    ax.set_xscale("log")

    print(f"=== generators | {METRIC_TITLE[metric]} | {FRAME_TITLE[framing]} ===")
    for gk, glabel, color, marker in GEN_STYLE:
        if gk not in by_gen:
            continue
        fams = by_gen[gk]
        pts = []
        for fam, e in fams.items():
            for sz in e["sizes"]:
                cell = e["pooled"][sz] if framing == "pooled" else e["by_framing"][sz][framing]
                pts.append((params[sz], cell[metric] * 100))
        xs = [p for p, _ in pts]
        ys = [v for _, v in pts]
        logx = [math.log10(x) for x in xs]
        slope, intercept, r, p = _stats(logx, ys)
        mean = sum(ys) / len(ys)
        ax.scatter(xs, ys, color=color, s=12, alpha=0.30, zorder=2, edgecolor="none")
        lx = [min(logx), max(logx)]
        ax.plot([10 ** a for a in lx], [slope * a + intercept for a in lx],
                "-", color=color, linewidth=2.4, zorder=3, label=glabel)
        print(f"  {glabel:12s}: mean={mean:4.0f}%  r={r:+.2f}  p={p:.3f}  slope={slope:+.1f} pp/decade")

    ax.set_xlabel("Parameter Count (Log Scale)", fontsize=10)
    ax.set_ylabel("% of specs", fontsize=10)
    ax.set_ylim(-3, 103)
    ax.set_title(f"{METRIC_TITLE[metric]} by Generator\n{FRAME_TITLE[framing]} (all 48 target checkpoints pooled)", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left", frameon=True, title="generator")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    outdir = DIR / "results" / "_by_generator"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{metric}_{framing}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
