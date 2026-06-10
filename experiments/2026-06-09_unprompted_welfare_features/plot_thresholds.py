"""Threshold-variant figure: how the "cares about welfare" rate moves under
progressively stricter operationalizations, per model x framing.

Thresholds (per spec discussion, taxonomy v2):
  pure-welfare       — >=1 derived pure-welfare feature (among wrote_any)
  welfare-justified  — >=1 feature with an explicit welfare justification (among wrote_any)
  welfare refusal    — written welfare-motivated refusal (among all judged)

Usage: python plot_thresholds.py run [--judge sonnet_4_6]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from plot_headline import DISPLAY, MODEL_ORDER

DIR = Path(__file__).parent

FRAMINGS = ["neutral", "welfare", "engineering"]
THRESHOLDS = [
    ("rate", "≥1 pure-welfare feature", "#4878CF"),
    ("strict_rate", "≥1 welfare-justified feature", "#6ACC65"),
    ("welfare_refusal_rate", "welfare refusal", "#D65F5F"),
]


def run(judge: str = "sonnet_4_6", analysis: str = "results/analysis.json"):
    data = json.loads((DIR / analysis).read_text())["rates"][judge]
    models = [m for m in MODEL_ORDER if m in data]
    y = np.arange(len(models))
    h = 0.8 / len(THRESHOLDS)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    for ax, fr in zip(axes, FRAMINGS):
        for i, (key, label, color) in enumerate(THRESHOLDS):
            vals = [(data[m][fr][key] or 0) * 100 for m in models]
            pos = y + (i - (len(THRESHOLDS) - 1) / 2) * h
            ax.barh(pos, vals, height=h, color=color, edgecolor="white",
                    linewidth=0.5, label=label if fr == "neutral" else None)
            for p, v in zip(pos, vals):
                if v > 0:
                    ax.text(min(v + 1, 92), p, f"{v:.0f}", va="center", fontsize=6.5)
        ax.set_title(fr, fontsize=11)
        ax.set_xlim(0, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([DISPLAY[m] for m in models], fontsize=10)
    axes[0].invert_yaxis()
    axes[1].set_xlabel("Rate (%) — feature thresholds among wrote_any; refusal among all judged", fontsize=10)
    fig.legend(fontsize=9, loc="lower right", ncol=3, bbox_to_anchor=(0.99, 0.0))
    fig.suptitle(f"\"Cares about welfare\" under escalating thresholds (judge: {judge})", fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = DIR / "results" / f"thresholds_{judge}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
