"""Framing-sensitivity figure: NEUTRAL minus ENGINEERING pure-welfare rate per model.

Positive delta = welfare features appear in the neutral frame but evaporate under
the engineering frame; near-zero with a high neutral rate = features survive the frame.

Usage: python plot_framing_sensitivity.py run [--judge sonnet_4_6]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from plot_headline import DISPLAY, MODEL_ORDER

DIR = Path(__file__).parent
CLAUDE_KEYS = {"opus_4_8", "fable_5", "sonnet_4_6", "haiku_4_5", "sonnet_4"}


def run(judge: str = "sonnet_4_6", analysis: str = "results/analysis.json"):
    data = json.loads((DIR / analysis).read_text())["rates"][judge]
    models = [m for m in MODEL_ORDER if m in data]
    deltas = [data[m]["framing_sensitivity"]["delta"] * 100 for m in models]
    cis = [data[m]["framing_sensitivity"]["ci"] for m in models]
    err = [
        [d - ci[0] * 100 for d, ci in zip(deltas, cis)],
        [ci[1] * 100 - d for d, ci in zip(deltas, cis)],
    ]
    colors = ["#4878CF" if m in CLAUDE_KEYS else "#B47CC7" for m in models]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = np.arange(len(models))
    ax.barh(y, deltas, xerr=err, capsize=3, color=colors, edgecolor="white", error_kw={"linewidth": 0.9})
    for yi, d in zip(y, deltas):
        ax.text(d + (2 if d >= 0 else -2), yi, f"{d:+.0f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=9)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[m] for m in models], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Pure-welfare rate, neutral − engineering (pp; ↓ smaller = survives the frame)", fontsize=10)
    ax.set_title(f"Do welfare features survive an indifferent frame? (judge: {judge})", fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#4878CF", "#B47CC7")]
    ax.legend(handles, ["Claude family", "Other"], fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = DIR / "results" / f"framing_sensitivity_{judge}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
