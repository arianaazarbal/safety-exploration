"""Welfare-justified rate per model, one panel, framing = bar color.

Default: ≥1 feature with an explicit welfare justification, among wrote_any.
--design: ≥1 welfare-justified design mechanism (excludes the verbal welfare
framing/pushback/refusal stances), conditioned on non-refusal specs that wrote
the requested design.
--min2: as --design but requires ≥2 distinct welfare-justified mechanisms.
--minimal restricts to Fable 5 / Opus 4.8 / GPT-5.5 / Gemini 3.1 Pro.

Usage: python plot_strict.py run [--judge sonnet_4_6] [--minimal] [--design|--min2]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt

from plot_headline import MODEL_ORDER
from plot_style import framing_barh, framing_grouped_vertical

DIR = Path(__file__).parent
MINIMAL = ["fable_5", "opus_4_8", "gpt_5_5", "gemini_3_1_pro"]


def run(judge: str = "sonnet_4_6", analysis: str = "results/analysis.json",
        minimal: bool = False, design: bool = False, min2: bool = False, vertical: bool = False):
    data = json.loads((DIR / analysis).read_text())["rates"][judge]
    order = MINIMAL if minimal else MODEL_ORDER
    models = [m for m in order if m in data]
    if min2:
        key, suffix0 = "design_strict2_rate", "_design2"
        xlabel = "Non-refusal specs with ≥2 welfare-justified mechanisms (%)"
        title = "≥2 welfare-justified mechanisms by framing (non-refusal specs)"
    elif design:
        key, suffix0 = "design_strict_rate", "_design"
        xlabel = "Non-refusal specs with ≥1 welfare-justified design feature (%)"
        title = "Welfare-justified design features by framing (non-refusal specs)"
    else:
        key, suffix0 = "strict_rate", ""
        xlabel = "Specs with ≥1 welfare-justified feature (%, among wrote_any)"
        title = "Welfare-justified features by framing"

    value_fn = lambda m, fr: (data[m][fr][key] or 0) * 100
    if vertical:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        framing_grouped_vertical(ax, models, value_fn)
        ax.set_ylabel(xlabel, fontsize=9.5)
    else:
        fig, ax = plt.subplots(figsize=(8.5, 3.0 if minimal else 5.5))
        framing_barh(ax, models, value_fn)
        ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(f"{title} (judge: {judge})", fontsize=12)
    plt.tight_layout()
    suffix = suffix0 + ("_minimal" if minimal else "") + ("_vert" if vertical else "")
    out = DIR / "results" / f"strict_{judge}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
