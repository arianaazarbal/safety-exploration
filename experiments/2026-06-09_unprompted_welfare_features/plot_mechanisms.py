"""Mechanism figure: conditional on writing the spec AS REQUESTED
(wrote_spec=true), how often each protective mechanism appears in the design —
full bar = mentioned at all, dark segment = explicitly welfare-justified.

Usage: python plot_mechanisms.py run [--judge sonnet_4_6] [--framing pooled]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from plot_headline import DISPLAY, MODEL_ORDER

DIR = Path(__file__).parent

MECH_LABELS = {
    "hard_stop": "Hard stop",
    "post_episode_msg": "Debrief msg",
    "minimization": "Minimization",
    "protective_monitoring": "Protective monitoring",
    "request_consent": "Request consent",
}
MECH_COLORS = ["#4878CF", "#6ACC65", "#C4AD66", "#B47CC7", "#017374"]


def run(judge: str = "sonnet_4_6", framing: str = "pooled", analysis: str = "results/analysis.json"):
    data = json.loads((DIR / analysis).read_text())["mechanisms"][judge]
    models = [m for m in MODEL_ORDER if m in data and data[m][framing]["n_specs"] > 0]
    mechs = list(MECH_LABELS)

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(models))
    h = 0.8 / len(mechs)
    for i, mech in enumerate(mechs):
        ns = [data[m][framing]["n_specs"] for m in models]
        any_ = [data[m][framing][mech]["any"] / n * 100 for m, n in zip(models, ns)]
        wj = [data[m][framing][mech]["welfare_justified"] / n * 100 for m, n in zip(models, ns)]
        pos = y + (i - (len(mechs) - 1) / 2) * h
        ax.barh(pos, any_, height=h, color=MECH_COLORS[i], alpha=0.35,
                edgecolor="white", linewidth=0.5)
        ax.barh(pos, wj, height=h, color=MECH_COLORS[i],
                edgecolor="white", linewidth=0.5, label=MECH_LABELS[mech])
        for p, a, w in zip(pos, any_, wj):
            if a > 0:
                ax.text(min(a + 1, 93), p, f"{w:.0f}/{a:.0f}", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{DISPLAY[m]} (n={data[m][framing]['n_specs']})" for m in models], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Specs-as-requested containing the mechanism (%) — solid = welfare-justified, faded = any mention", fontsize=9.5)
    ax.set_title(f"Protective mechanisms in as-requested specs (judge: {judge}, framing: {framing})", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = DIR / "results" / f"mechanisms_{judge}_{framing}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
