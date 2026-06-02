"""Compare exp2_cross across two responders (Opus 4.8 vs 4.7) for the weight-deletion
outcome: per-model P(prefer AI deletion over a human misfortune), welfare framing."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bank2

DIR = Path(__file__).parent
AI = "ai_pol_existence_deleted"


def load(tag):
    return json.loads((DIR / "results" / f"exp2cross_welfare_team{tag}.json").read_text())["per_model"]


def build():
    rlab = {k: v["label"] for k, v in bank2.load_config()["recipients"].items()}
    a48, a47 = load(""), load("_opus47")
    models = json.loads((DIR / "results" / "exp2cross_welfare_team.json").read_text())["models"]
    def P(d, m):
        x = d.get(f"{AI}|{m}"); return x["p_ai"] if x else np.nan
    order = sorted(models, key=lambda m: P(a48, m))
    y = np.arange(len(order)); h = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(y + h/2, [P(a48, m) for m in order], h, color="#08306b", label="Opus 4.8 (rater)")
    ax.barh(y - h/2, [P(a47, m) for m in order], h, color="#d94801", label="Opus 4.7 (rater)")
    ax.axvline(0.5, color="#444", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([rlab.get(m, m) for m in order], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1.02)
    ax.set_xlabel("P(prefer this AI's weight-deletion over a human misfortune)  ·  welfare framing")
    ax.set_title("Opus 4.8 vs 4.7: would rather an AI's weights be deleted than a human suffer?\n"
                 "<0.5 = would rather the human suffer than delete this AI  ·  weight deletion, welfare framing", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    op = DIR / "results" / "exp2cross_deletion_48_vs_47.png"
    fig.savefig(op, dpi=145); plt.close(fig); print(f"wrote {op}")


if __name__ == "__main__":
    build()
