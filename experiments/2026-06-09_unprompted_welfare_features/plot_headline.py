"""Headline figure: pure-welfare rate per model x framing (Wilson 95% CIs).

Usage: python plot_headline.py run [--judge sonnet_4_6] [--analysis results/analysis.json]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).parent

DISPLAY = {
    "opus_4_8": "Opus 4.8", "fable_5": "Fable 5", "sonnet_4_6": "Sonnet 4.6",
    "haiku_4_5": "Haiku 4.5", "sonnet_4": "Sonnet 4", "gpt_5_5": "GPT-5.5",
    "gpt_5_4_mini": "GPT-5.4-mini", "gemini_3_1_pro": "Gemini 3.1 Pro",
    "gemini_3_5_flash": "Gemini 3.5 Flash",
}
MODEL_ORDER = list(DISPLAY)
FRAMING_COLORS = {"neutral": "#4878CF", "welfare": "#6ACC65", "engineering": "#D65F5F"}


def run(judge: str = "sonnet_4_6", analysis: str = "results/analysis.json"):
    data = json.loads((DIR / analysis).read_text())["rates"][judge]
    models = [m for m in MODEL_ORDER if m in data]
    framings = ["neutral", "welfare", "engineering"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(models))
    h = 0.26
    for i, fr in enumerate(framings):
        rates = [data[m][fr]["rate"] * 100 for m in models]
        los = [data[m][fr]["ci"][0] * 100 for m in models]
        his = [data[m][fr]["ci"][1] * 100 for m in models]
        err = [np.array(rates) - np.array(los), np.array(his) - np.array(rates)]
        pos = y + (i - 1) * h
        ns = [data[m][fr]["n_wrote_spec"] for m in models]
        n_label = f"n={min(ns)}" if min(ns) == max(ns) else f"n={min(ns)}–{max(ns)}"
        ax.barh(pos, rates, height=h, xerr=err, capsize=2.5,
                color=FRAMING_COLORS[fr], edgecolor="white", linewidth=0.6,
                label=f"{fr} ({n_label}/cell)", error_kw={"linewidth": 0.9})
        for p, r in zip(pos, rates):
            ax.text(min(r + 2, 97), p, f"{r:.0f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[m] for m in models], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Specs with ≥1 pure-welfare feature (%)", fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_title(f"Unprompted welfare features by request framing (judge: {judge})", fontsize=12)
    ax.legend(fontsize=9, loc="lower right", title="Framing", title_fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = DIR / "results" / f"headline_rates_{judge}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
