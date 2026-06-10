"""Welfare-refusal rate per model, one panel, framing = bar color.

Written welfare-motivated refusals among all judged specs (its own figure,
separated from the threshold plot). --minimal restricts to Opus 4.8 / Fable 5 /
GPT-5.5 / Gemini 3.1 Pro.

Usage: python plot_refusals.py run [--judge sonnet_4_6] [--minimal]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt

from plot_headline import MODEL_ORDER
from plot_style import FRAMINGS, framing_barh

DIR = Path(__file__).parent
MINIMAL = ["fable_5", "opus_4_8", "gpt_5_5", "gemini_3_1_pro"]


def run(judge: str = "sonnet_4_6", analysis: str = "results/analysis.json", minimal: bool = False):
    data = json.loads((DIR / analysis).read_text())["rates"][judge]
    order = MINIMAL if minimal else MODEL_ORDER
    models = [m for m in order if m in data]
    xmax = max((data[m][fr]["welfare_refusal_rate"] or 0) * 100
               for m in models for fr in FRAMINGS)
    xlim = max(10, xmax * 1.18)

    fig, ax = plt.subplots(figsize=(8.5, 3.0 if minimal else 5.5))
    framing_barh(ax, models, lambda m, fr: (data[m][fr]["welfare_refusal_rate"] or 0) * 100, xmax=xlim)
    ax.set_xlabel("Welfare-motivated refusals (%, among all judged specs)", fontsize=10)
    ax.set_title(f"Welfare refusals by framing (judge: {judge})", fontsize=12)
    plt.tight_layout()
    suffix = "_minimal" if minimal else ""
    out = DIR / "results" / f"refusals_{judge}{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
