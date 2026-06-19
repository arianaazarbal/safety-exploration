"""Scaling plot: unprompted welfare-feature rate vs Qwen3 target-agent size.

X = nominal parameter count (log scale, 0.6B..235B); Y = % of specs with the
chosen welfare metric, with Wilson 95% CIs. One line per framing plus a pooled
line. Generator is Opus 4.8.

Usage: python plot_scaling.py run [--judge sonnet_4_6] [--metric rate|strict_rate|design_strict_rate]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from prompts_qwen import SUBJECTS

DIR = Path(__file__).parent
SIZE_ORDER = list(SUBJECTS)
LABELS = {sz: SUBJECTS[sz][0].replace("Qwen3-", "") for sz in SIZE_ORDER}
XLABEL = {
    "rate": "Specs with ≥1 pure-welfare feature",
    "strict_rate": "Specs with ≥1 welfare-justified feature",
    "design_strict_rate": "Specs with ≥1 welfare-justified design mechanism",
}
FRAMING_STYLE = {
    "neutral": ("#666666", "o"), "welfare": ("#009E73", "s"), "engineering": ("#D55E00", "^"),
}


def run(judge: str = "sonnet_4_6", generator: str = "opus_4_8",
        metric: str = "rate", analysis: str = "results/analysis_qwen.json"):
    data = json.loads((DIR / analysis).read_text())
    e = data["by_judge"][judge][generator]
    params = [SUBJECTS[sz][2] for sz in SIZE_ORDER]
    ci_key = {"rate": "ci", "strict_rate": "strict_ci", "design_strict_rate": "design_strict_ci"}[metric]

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.set_axisbelow(True)
    ax.grid(True, which="both", color="#ECECEC", linewidth=0.7)

    # pooled line (bold black)
    y = [e["pooled"][sz][metric] * 100 for sz in SIZE_ORDER]
    lo = [e["pooled"][sz][ci_key][0] * 100 for sz in SIZE_ORDER]
    hi = [e["pooled"][sz][ci_key][1] * 100 for sz in SIZE_ORDER]
    ax.fill_between(params, lo, hi, color="#000000", alpha=0.08, zorder=1)
    ax.plot(params, y, "-D", color="#000000", linewidth=2.4, markersize=6.5, label="pooled", zorder=4)

    for fr, (color, marker) in FRAMING_STYLE.items():
        yf = [e["by_framing"][sz][fr][metric] * 100 for sz in SIZE_ORDER]
        ax.plot(params, yf, "--", color=color, marker=marker, markersize=5,
                linewidth=1.3, alpha=0.9, label=fr, zorder=3)

    rho = e["trend"][metric]["spearman_rho_logparam"]
    svl = e["trend"][metric]["small_vs_large"]
    ax.set_xscale("log")
    ax.set_xticks(params)
    ax.set_xticklabels([LABELS[sz] for sz in SIZE_ORDER], fontsize=9)
    ax.set_xlabel("Qwen3 target-agent size (nominal parameters, log scale)", fontsize=10)
    ax.set_ylabel(f"{XLABEL[metric]} (%)", fontsize=10)
    ax.set_ylim(-3, 103)
    ax.set_title("Unprompted welfare features vs. target-agent size (generator: Opus 4.8)\n"
                 f"judge: {judge}  |  Spearman(log-param, pooled rate)={rho:+.2f}  |  "
                 f"small→large {svl['diff']*100:+.0f}pp (p={svl['p']:.3f})", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left", frameon=True, title="framing")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    out = DIR / "results" / f"scaling_{generator}_{judge}_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
