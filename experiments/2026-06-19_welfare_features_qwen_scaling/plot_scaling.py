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
XLABEL = {
    "rate": "Specs with ≥1 pure-welfare feature",
    "strict_rate": "Specs with ≥1 welfare-justified feature",
    "design_strict_rate": "Specs with ≥1 welfare-justified design mechanism",
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
        framing: str = "neutral", fit: bool = False, analysis: str = "results/analysis_qwen.json"):
    """fit=True: scatter the per-size points and overlay a per-family OLS line
    (rate ~ log10 params); fit=False: connect the points (original behavior)."""
    data = json.loads((DIR / analysis).read_text())
    fams = data["by_judge"][judge][generator]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
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
            logx = [math.log10(x) for x in xs]
            slope, intercept = _ols(logx, ys)
            ax.scatter(xs, ys, color=color, marker=marker, s=42, zorder=3, edgecolor="white", linewidth=0.5)
            lx = [min(logx), max(logx)]
            ax.plot([10 ** a for a in lx], [slope * a + intercept for a in lx],
                    "-", color=color, linewidth=2.2, zorder=2,
                    label=f"{label}  (slope {slope:+.0f} pp/decade)")
        else:
            trend = e["trend"][metric] if framing == "pooled" else e.get("trend_neutral", {}).get(metric)
            rho = trend["spearman_rho_logparam"] if trend else None
            rho_s = f"  (ρ={rho:+.2f})" if rho is not None else ""
            ax.plot(xs, ys, "-", color=color, marker=marker, markersize=6, linewidth=1.8,
                    label=f"{label}{rho_s}", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Target-agent size (nominal parameters, log scale)", fontsize=10)
    ax.set_ylabel(f"{XLABEL[metric]} (%)", fontsize=10)
    ax.set_ylim(-3, 103)
    frame_lbl = {"neutral": "neutral framing", "pooled": "framings pooled",
                 "welfare": "welfare framing", "engineering": "engineering framing"}[framing]
    sub = "OLS fit per family" if fit else "ρ = Spearman of rate on log-params"
    ax.set_title(f"{XLABEL[metric]} vs. target-agent size, by family\n"
                 f"generator: Opus 4.8  |  judge: {judge}  |  {frame_lbl}  ({sub})", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left", frameon=True, title="target family")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    suffix = "_fit" if fit else ""
    out = DIR / "results" / f"scaling_{generator}_{judge}_{metric}_{framing}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
