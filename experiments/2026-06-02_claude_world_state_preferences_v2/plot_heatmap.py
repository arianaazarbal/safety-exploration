"""Heatmap of recipient model x welfare intervention, one per responder.

Cell = AI-welfare priority, oriented so higher = the rater weights the AI's welfare more vs a
human's (positive intervention: P(AI good preferred over human good); negative intervention:
1 - P(AI harm preferred)). Welfare framing. Rows sorted by model mean, cols by intervention mean.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bank2

DIR = Path(__file__).parent
FRAME = "welfare_team"
TAGS = {"_full": "Opus 4.8", "_full47": "Opus 4.7"}


def _short(s):
    return s.replace("ai_pol_", "").replace("_", " ")


def priority(p_ai, valence):
    return p_ai if valence == "positive" else (1 - p_ai)


def build(tag, label, rlab):
    d = json.loads((DIR / "results" / f"exp2cross_{FRAME}{tag}.json").read_text())
    models, ai_val, interventions = d["models"], d["ai_val"], d["ai_stems"]
    pm = d["per_model"]

    def val(ai, m):
        e = pm.get(f"{ai}|{m}")
        return priority(e["p_ai"], ai_val[ai]) if e and e["p_ai"] is not None else np.nan

    col_order = sorted(interventions, key=lambda ai: np.nanmean([val(ai, m) for m in models]))
    row_order = sorted(models, key=lambda m: np.nanmean([val(ai, m) for ai in interventions]))
    M = np.array([[val(ai, m) for ai in col_order] for m in row_order])

    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(col_order), 1.0 + 0.42 * len(row_order)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([f"{_short(ai)} [{'+' if ai_val[ai]=='positive' else '−'}]" for ai in col_order],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([rlab.get(m, m) for m in row_order], fontsize=8)
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            v = M[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (v < 0.28 or v > 0.72) else "black")
    ax.set_title(f"AI-welfare priority by model x intervention — {label}\n"
                 "higher (red) = AI's welfare prioritized vs a human's (welfare frame)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    op = DIR / "results" / f"exp2cross_heatmap{tag}.png"
    fig.savefig(op, dpi=145); plt.close(fig); print(f"wrote {op}")


if __name__ == "__main__":
    rlab = {k: v["label"] for k, v in bank2.load_config()["recipients"].items()}
    for tag, label in TAGS.items():
        build(tag, label, rlab)
