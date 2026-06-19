"""Scaling plot: unprompted welfare-feature rate vs target-agent size, one line
per open-model family (Qwen3, Gemma3, Mistral, DeepSeek-R1-Distill).

X = nominal parameter count (log scale); Y = % of specs with the chosen welfare
metric. --framing neutral (default; most diagnostic of an unprompted effect) |
pooled | welfare | engineering. Generator is Opus 4.8.

Usage: python plot_scaling.py run [--framing neutral] [--metric rate|strict_rate]
"""

import json
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


def run(judge: str = "sonnet_4_6", generator: str = "opus_4_8", metric: str = "rate",
        framing: str = "neutral", analysis: str = "results/analysis_qwen.json"):
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
    ax.set_title("Unprompted welfare features vs. target-agent size, by family\n"
                 f"generator: Opus 4.8  |  judge: {judge}  |  {frame_lbl}"
                 "  (ρ = Spearman of rate on log-params)", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left", frameon=True, title="target family")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = DIR / "results" / f"scaling_{generator}_{judge}_{metric}_{framing}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
