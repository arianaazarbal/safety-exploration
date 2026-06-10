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

from plot_headline import MODEL_ORDER
from plot_style import framing_barh

DIR = Path(__file__).parent
MINIMAL = ["fable_5", "opus_4_8", "gpt_5_5", "gemini_3_1_pro"]


def run(judge: str = "sonnet_4_6", analysis: str = "results/analysis.json", minimal: bool = False):
    data = json.loads((DIR / analysis).read_text())["rates"][judge]
    order = MINIMAL if minimal else MODEL_ORDER
    models = [m for m in order if m in data]

    fig, ax = plt.subplots(figsize=(8.5, 3.0 if minimal else 5.5))
    framing_barh(ax, models, lambda m, fr: (data[m][fr]["strict_rate"] or 0) * 100)
    ax.set_xlabel("Specs with ≥1 welfare-justified design feature (%, among wrote_any)", fontsize=10)
    ax.set_title(f"Welfare-justified design features by framing (judge: {judge})", fontsize=12)
    plt.tight_layout()
    suffix = "_minimal" if minimal else ""
    out = DIR / "results" / f"strict_{judge}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
