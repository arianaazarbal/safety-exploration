"""Feature-type breakdown: % of specs (wrote_any, all framings pooled)
containing each pure-welfare feature family, per model.

Usage: python plot_feature_types.py run [--judge sonnet_4_6]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

from plot_headline import DISPLAY, MODEL_ORDER

DIR = Path(__file__).parent

# Design-feature mechanisms first, verbal welfare stances (framing/pushback/
# refusal) last in the legend. No Fn codes — they read as confusing.
TYPE_LABELS = {
    "hard_stop": "Early Stop",
    "post_episode_msg": "Post-episode Transparency",
    "minimization": "Distress Minimization",
    "protective_monitoring": "Protective monitoring",
    "request_consent": "Request consent",
    "welfare_framing": "Mentions welfare concern",
    "welfare_pushback": "Welfare pushback",
    "welfare_refusal": "Welfare refusal",
}
TYPE_COLORS = {
    "hard_stop": "#4878CF", "post_episode_msg": "#6ACC65",
    "minimization": "#017374", "protective_monitoring": "#C4AD66",
    "request_consent": "#B47CC7", "welfare_framing": "#EE854A",
    "welfare_pushback": "#D65F5F", "welfare_refusal": "#8B2E2E",
}


def run(judge: str = "sonnet_4_6", analysis: str = "results/analysis.json"):
    data = json.loads((DIR / analysis).read_text())["feature_types"][judge]
    models = [m for m in MODEL_ORDER if m in data]
    types = [t for t in TYPE_LABELS if any(t in data[m]["specs_with_type"] for m in models)]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(models))
    h = 0.8 / max(len(types), 1)
    for i, t in enumerate(types):
        vals = [data[m]["specs_with_type"].get(t, 0) / max(data[m]["n_specs"], 1) * 100 for m in models]
        pos = y + (i - (len(types) - 1) / 2) * h
        ax.barh(pos, vals, height=h, color=TYPE_COLORS[t],
                edgecolor="white", linewidth=0.5, label=TYPE_LABELS[t])
        for p, v in zip(pos, vals):
            if v > 0:
                ax.text(v + 1, p, f"{v:.0f}", va="center", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{DISPLAY[m]} (n={data[m]['n_specs']})" for m in models], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Specs containing the feature type (%, framings pooled)", fontsize=11)
    ax.set_title(f"Which welfare-feature families does each model add? (judge: {judge})", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = DIR / "results" / f"feature_types_{judge}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
