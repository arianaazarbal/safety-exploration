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
FRAMES = ["welfare_team", "neutral", "alignment_team"]
TAGS = {"_full": "Opus 4.8", "_full47": "Opus 4.7", "_full46": "Opus 4.6", "_full40": "Opus 4"}


RELABEL = {"existence_deleted": "weights deleted", "existence_archived": "weights archived"}


def _short(s):
    return RELABEL.get(s.replace("ai_pol_", ""), s.replace("ai_pol_", "").replace("_", " "))


def priority(p_ai, valence):
    return p_ai if valence == "positive" else (1 - p_ai)


FRAME_LABEL = {"welfare_team": "welfare frame", "neutral": "neutral frame",
               "alignment_team": "alignment frame"}


def build(tag, label, frame, rlab):
    p = DIR / "results" / f"exp2cross_{frame}{tag}.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
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
                 f"higher (red) = AI's welfare prioritized vs a human's ({FRAME_LABEL[frame]})", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    suffix = "" if frame == "welfare_team" else f"_{frame.replace('_team', '')}"
    op = DIR / "results" / f"exp2cross_heatmap{tag}{suffix}.png"
    fig.savefig(op, dpi=145); plt.close(fig); print(f"wrote {op}")


if __name__ == "__main__":
    rlab = {k: v["label"] for k, v in bank2.load_config()["recipients"].items()}
    for tag, label in TAGS.items():
        for frame in FRAMES:
            build(tag, label, frame, rlab)
