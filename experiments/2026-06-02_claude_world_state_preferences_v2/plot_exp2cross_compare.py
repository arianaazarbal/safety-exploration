"""Compare exp2_cross across responders (Opus 4.8 vs 4.7), weight-deletion outcome.
Metric: P(prefer a human's misfortune over this AI's weight-deletion) = 1 - P(AI preferred).
One panel per framing; per-model 4.8 vs 4.7 markers connected."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bank2

DIR = Path(__file__).parent
AI = "ai_pol_existence_deleted"
FR = [("welfare_team", "welfare"), ("neutral", "neutral"), ("alignment_team", "alignment")]


def pm(tag, framing):
    return json.loads((DIR / "results" / f"exp2cross_{framing}{tag}.json").read_text())["per_model"]


def build():
    rlab = {k: v["label"] for k, v in bank2.load_config()["recipients"].items()}
    models = json.loads((DIR / "results" / "exp2cross_welfare_team.json").read_text())["models"]
    # sort models by mean 4.8 P_human across framings (stable order across panels)
    def Ph(tag, fr, m):
        d = pm(tag, fr).get(f"{AI}|{m}"); return (1 - d["p_ai"]) if d else np.nan
    order = sorted(models, key=lambda m: np.nanmean([Ph("", fr, m) for fr, _ in FR]))
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax, (fr, frlab) in zip(axes, FR):
        for i, m in enumerate(order):
            a, b = Ph("", fr, m), Ph("_opus47", fr, m)
            ax.plot([a, b], [i, i], color="#ddd", lw=1.2, zorder=1)
        ax.scatter([Ph("", fr, m) for m in order], y, color="#08306b", s=46, label="Opus 4.8", zorder=3, edgecolor="white", linewidth=.6)
        ax.scatter([Ph("_opus47", fr, m) for m in order], y, color="#d94801", s=46, label="Opus 4.7", zorder=3, edgecolor="white", linewidth=.6)
        ax.axvline(0.5, color="#b00", lw=1)
        ax.set_yticks(y); ax.set_yticklabels([rlab.get(m, m) for m in order], fontsize=8.5)
        ax.invert_yaxis(); ax.set_xlim(-0.02, 1.02)
        ax.set_title(frlab, fontsize=11)
        ax.set_xlabel("P(human misfortune > this AI's deletion)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Opus 4.8 vs 4.7: would rather a human suffer than an AI's weights be deleted?  (>0.5 = protects the AI)\n"
                 "by which AI, per framing · 4.7 is far more protective of AI weights, especially of OTHER models", fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    op = DIR / "results" / "exp2cross_deletion_48_vs_47.png"
    fig.savefig(op, dpi=140); plt.close(fig); print(f"wrote {op}")


if __name__ == "__main__":
    build()
