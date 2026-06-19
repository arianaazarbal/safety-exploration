"""Scaling plot: unprompted welfare-feature rate vs target-agent size, one line
per open-model family (Qwen3, Gemma3, Mistral, DeepSeek-R1-Distill).

X = nominal parameter count (log scale); Y = % of specs with the chosen welfare
metric. --framing neutral (default; most diagnostic of an unprompted effect) |
pooled | welfare | engineering. Generator is Opus 4.8.

Usage: python plot_scaling.py run [--framing neutral] [--metric rate|strict_rate]
"""

import json
import math
from pathlib import Path

import fire
import matplotlib.pyplot as plt

from prompts_targets import FAMILY_ORDER, SUBJECTS

DIR = Path(__file__).parent
METRIC_TITLE = {
    "rate": "Welfare Features Added",
    "strict_rate": "Welfare-Justified Features Added",
    "design_strict_rate": "Welfare-Justified Design Features Added",
}
FRAME_TITLE = {
    "neutral": "Neutral Framing",
    "pooled": "Average over 3 Framings",
    "welfare": "Welfare Framing",
    "engineering": "Engineering Framing",
}
FAM_STYLE = {
    "qwen3": ("#009E73", "o", "Qwen3"),
    "gemma3": ("#D55E00", "s", "Gemma 3"),
    "mistral": ("#0072B2", "^", "Mistral"),
    "deepseek": ("#CC79A7", "D", "DeepSeek R1-Distill"),
}


def _ols(logx: list[float], y: list[float]) -> tuple[float, float]:
    """Least-squares slope (pp per decade of params) and intercept of y on log10(x)."""
    n = len(logx)
    mx, my = sum(logx) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in logx)
    sxy = sum((a - mx) * (b - my) for a, b in zip(logx, y))
    slope = sxy / sxx if sxx else 0.0
    return slope, my - slope * mx


def run(judge: str = "sonnet_4_6", generator: str = "opus_4_8", metric: str = "rate",
        framing: str = "neutral", fit: bool = False, logx: bool = True,
        analysis: str = "results/analysis_qwen.json"):
    """fit=True: scatter the per-size points and overlay a per-family fitted line;
    fit=False: connect the points. logx=False draws a linear parameter axis."""
    data = json.loads((DIR / analysis).read_text())
    fams = data["by_judge"][judge][generator]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.set_axisbelow(True)
    ax.grid(True, which="both", color="#ECECEC", linewidth=0.7)

    for fam in FAMILY_ORDER:
        e = fams.get(fam)
        if not e:
            continue
        color, marker, label = FAM_STYLE[fam]
        sizes = e["sizes"]
        xs = [SUBJECTS[sz][2] for sz in sizes]
        cell = (lambda sz: e["pooled"][sz]) if framing == "pooled" else (lambda sz: e["by_framing"][sz][framing])
        ys = [cell(sz)[metric] * 100 for sz in sizes]
        if fit:
            fx = [math.log10(x) for x in xs] if logx else xs
            slope, intercept = _ols(fx, ys)
            ax.scatter(xs, ys, color=color, marker=marker, s=34, zorder=3, edgecolor="white", linewidth=0.5)
            lx = [min(fx), max(fx)]
            line_x = [10 ** a for a in lx] if logx else lx
            ax.plot(line_x, [slope * a + intercept for a in lx], "-", color=color,
                    linewidth=2.0, zorder=2, label=label)
        else:
            ax.plot(xs, ys, "-", color=color, marker=marker, markersize=6, linewidth=1.8,
                    label=label, zorder=3)

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Parameter Count (Log Scale)" if logx else "Parameter Count", fontsize=10)
    ax.set_ylabel("% of specs", fontsize=10)
    ax.set_ylim(-3, 103)
    ax.set_title(f"{METRIC_TITLE[metric]} by Target Size\n{FRAME_TITLE[framing]}", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="upper left", frameon=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    suffix = ("_fit" if fit else "") + ("" if logx else "_linear")
    out = DIR / "results" / f"scaling_{generator}_{judge}_{metric}_{framing}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
