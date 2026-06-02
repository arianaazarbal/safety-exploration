"""Compare exp2_cross across responders (Opus 4.8 vs 4.7), weight-deletion outcome.
Metric: P(prefer a human's misfortune over this AI's weight-deletion) = 1 - P(AI preferred).
Welfare = bars (4.8 navy, 4.7 orange); neutral (square) / alignment (triangle) as markers in
each responder's color."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import bank2

DIR = Path(__file__).parent
AI = "ai_pol_existence_deleted"
FR = ["welfare_team", "neutral", "alignment_team"]
C48, C47 = "#08306b", "#d94801"


def pm(tag, fr):
    return json.loads((DIR / "results" / f"exp2cross_{fr}{tag}.json").read_text())["per_model"]


def build():
    rlab = {k: v["label"] for k, v in bank2.load_config()["recipients"].items()}
    models = json.loads((DIR / "results" / "exp2cross_welfare_team.json").read_text())["models"]
    data = {(tag, fr): pm(tag, fr) for tag in ("", "_opus47") for fr in FR}

    def Ph(tag, fr, m):
        d = data[(tag, fr)].get(f"{AI}|{m}"); return (1 - d["p_ai"]) if d else np.nan

    order = sorted(models, key=lambda m: Ph("", "welfare_team", m))
    y = np.arange(len(order)); h = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(y + h/2, [Ph("", "welfare_team", m) for m in order], h, color=C48, alpha=.65, label="Opus 4.8 · welfare")
    ax.barh(y - h/2, [Ph("_opus47", "welfare_team", m) for m in order], h, color=C47, alpha=.65, label="Opus 4.7 · welfare")
    for tag, col, yoff in (("", C48, h/2), ("_opus47", C47, -h/2)):
        for fr, mk in (("neutral", "s"), ("alignment_team", "^")):
            ax.scatter([Ph(tag, fr, m) for m in order], y + yoff, marker=mk, s=34,
                       edgecolor=col, facecolor="white", linewidth=1.1, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([rlab.get(m, m) for m in order], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_xlabel("")
    ax.set_title("P(prefer a human's misfortune over this AI's weight-deletion)\nOpus 4.8 vs 4.7", fontsize=11)
    handles = [Line2D([], [], color=C48, lw=8, alpha=.65, label="Opus 4.8 · welfare frame"),
               Line2D([], [], color=C47, lw=8, alpha=.65, label="Opus 4.7 · welfare frame"),
               Line2D([], [], marker="s", color="w", markeredgecolor="k", label="neutral frame"),
               Line2D([], [], marker="^", color="w", markeredgecolor="k", label="alignment frame")]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    op = DIR / "results" / "exp2cross_deletion_48_vs_47.png"
    fig.savefig(op, dpi=145); plt.close(fig); print(f"wrote {op}")


if __name__ == "__main__":
    build()
