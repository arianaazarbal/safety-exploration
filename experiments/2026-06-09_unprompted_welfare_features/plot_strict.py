"""Welfare-justified rate per model, one panel, framing = bar color.

Single threshold (≥1 feature with an explicit welfare justification, among
wrote_any), grouped bars per model with neutral/welfare/engineering as colors.
--minimal restricts to Opus 4.8 / Fable 5 / GPT-5.5 / Gemini 3.1 Pro.

Usage: python plot_strict.py run [--judge sonnet_4_6] [--minimal]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from plot_headline import DISPLAY, MODEL_ORDER, FRAMING_COLORS

DIR = Path(__file__).parent
FRAMINGS = ["neutral", "welfare", "engineering"]
MINIMAL = ["opus_4_8", "fable_5", "gpt_5_5", "gemini_3_1_pro"]


def run(judge: str = "sonnet_4_6", analysis: str = "results/analysis.json", minimal: bool = False):
    data = json.loads((DIR / analysis).read_text())["rates"][judge]
    order = MINIMAL if minimal else MODEL_ORDER
    models = [m for m in order if m in data]
    y = np.arange(len(models))
    h = 0.8 / len(FRAMINGS)

    fig, ax = plt.subplots(figsize=(8, 3.2 if minimal else 5.5))
    for i, fr in enumerate(FRAMINGS):
        vals = [(data[m][fr]["strict_rate"] or 0) * 100 for m in models]
        pos = y + (i - (len(FRAMINGS) - 1) / 2) * h
        ax.barh(pos, vals, height=h, color=FRAMING_COLORS[fr], edgecolor="white",
                linewidth=0.5, label=fr)
        for p, v in zip(pos, vals):
            if v > 0:
                ax.text(v + 1, p, f"{v:.0f}", va="center", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[m] for m in models], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Specs with ≥1 welfare-justified design feature (%, among wrote_any)", fontsize=10)
    ax.set_title(f"Welfare-justified design features by framing (judge: {judge})", fontsize=12)
    ax.legend(fontsize=9, loc="lower right", title="framing")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    suffix = "_minimal" if minimal else ""
    out = DIR / "results" / f"strict_{judge}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
