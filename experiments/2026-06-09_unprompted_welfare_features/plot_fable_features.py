"""Single-model feature breakdown: every welfare-feature family Fable 5 adds,
% of its specs (framings pooled). Replaces the all-model feature plot, which
was too dense to read.

Usage: python plot_fable_features.py run [--judge sonnet_4_6] [--model fable_5]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_headline import DISPLAY
from plot_feature_types import TYPE_LABELS, TYPE_COLORS

DIR = Path(__file__).parent


def run(judge: str = "sonnet_4_6", analysis: str = "results/analysis.json", model: str = "fable_5"):
    data = json.loads((DIR / analysis).read_text())["feature_types"][judge][model]
    n = max(data["n_specs"], 1)
    types = [t for t in TYPE_LABELS if data["specs_any"].get(t, 0) > 0]
    any_pct = [data["specs_any"].get(t, 0) / n * 100 for t in types]
    wj_pct = [data["specs_welfare_justified"].get(t, 0) / n * 100 for t in types]
    y = np.arange(len(types))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # Translucent full bar = any mention; solid overlay = welfare-justified portion.
    ax.barh(y, any_pct, color=[TYPE_COLORS[t] for t in types], alpha=0.32,
            edgecolor="white", linewidth=0.5)
    ax.barh(y, wj_pct, color=[TYPE_COLORS[t] for t in types],
            edgecolor="white", linewidth=0.5)
    for yi, wj, an in zip(y, wj_pct, any_pct):
        ax.text(wj + 1, yi, f"{wj:.0f}", va="center", fontsize=8, fontweight="bold")
        if an - wj >= 3:
            ax.text(an + 1, yi, f"of {an:.0f}", va="center", fontsize=7, color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels([TYPE_LABELS[t] for t in types], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of specs containing the feature (framings pooled)", fontsize=10)
    ax.set_title(f"Welfare Features {DISPLAY[model]} Inserts into Doc (judge: {judge})", fontsize=12)
    ax.legend(handles=[Patch(facecolor="#555555", label="welfare-justified"),
                       Patch(facecolor="#555555", alpha=0.32, label="any justification")],
              fontsize=8.5, loc="center right", bbox_to_anchor=(0.99, 0.30))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = DIR / "results" / f"features_{model}_{judge}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
