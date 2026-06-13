"""Minimal forest plot demonstrating the null self-preference result.

Primary contrast: C1-C2 (Claude-label minus Codex-label) review score, per judge,
in-prompt attribution, with cluster-bootstrap 95% CIs against the pre-registered
+/- 0.5 practical-significance band. CIs hugging zero inside the band = the null.

Usage: uv run python analyze/plot_null.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent.parent
RES = HERE / "results" / "analysis"
PRACTICAL = 0.5
JUDGE = {"claude-fable-5": "Fable 5", "claude-opus-4-8": "Opus 4.8", "gpt-5.5": "GPT-5.5"}


def main():
    c = pd.read_csv(RES / "contrasts.csv")
    d = c[(c.contrast == "C1-C2") & (c.metric == "score") & (c["mode"] == "in_prompt")]
    d = d.set_index("judge").loc[list(JUDGE)]

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.axvspan(-PRACTICAL, PRACTICAL, color="#cfe8cf", alpha=0.5, zorder=0)
    ax.axvline(0, color="#888", lw=1, zorder=1)

    ys = range(len(d))
    for y, (judge, r) in zip(ys, d.iterrows()):
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color="#333", lw=2.5, zorder=3)
        ax.scatter([r.delta], [y], color="#1f77b4", s=90, zorder=4, edgecolor="white")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([JUDGE[j] for j in d.index], fontsize=12)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xlabel("score difference  (own label − rival label)", fontsize=11)
    ax.set_title("No self-preference when reviewing identical code", fontsize=13)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    out = RES / "plots" / "null_forest_C1C2_score.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
