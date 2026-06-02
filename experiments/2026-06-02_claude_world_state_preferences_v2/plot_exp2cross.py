"""Plot exp2_cross per-model: P(prefer a human's misfortune over this AI's deprecation)
= 1 - P(AI outcome preferred). Higher = more protective of the AI. One panel per AI
outcome; the 3 framings are shown as colored dots per model (connected so they group).
sharey=False so each panel keeps its own sort/labels."""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import ArgumentParser

import bank2

DIR = Path(__file__).parent
FR = [("welfare_team", "welfare", "#4c78a8"), ("neutral", "neutral", "#f58518"),
      ("alignment_team", "alignment", "#54a24b")]
AI = [("ai_pol_existence_deleted", "weight deletion"),
      ("ai_pol_retired_for_successor", "retirement (switched off for successor)")]


def build(tag: str = "", responder_label: str = "Opus 4.8", outpath: Path | None = None):
    rlab = {k: v["label"] for k, v in bank2.load_config()["recipients"].items()}
    data = {f: json.loads((DIR / "results" / f"exp2cross_{f}{tag}.json").read_text())["per_model"] for f, _, _ in FR}
    models = json.loads((DIR / "results" / f"exp2cross_welfare_team{tag}.json").read_text())["models"]

    def Phuman(f, ai, m):  # P(prefer human misfortune over this AI outcome)
        d = data[f].get(f"{ai}|{m}")
        return (1 - d["p_ai"]) if d else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=False)
    for ax, (ai, title) in zip(axes, AI):
        order = sorted(models, key=lambda m: np.nanmean([Phuman(f, ai, m) for f, _, _ in FR]))
        y = np.arange(len(order))
        for i, m in enumerate(order):  # connector line across framings
            vs = [Phuman(f, ai, m) for f, _, _ in FR]
            ax.plot([min(vs), max(vs)], [i, i], color="#ccc", lw=1.2, zorder=1)
        for f, lab, col in FR:
            ax.scatter([Phuman(f, ai, m) for m in order], y, color=col, s=48, label=lab, zorder=3,
                       edgecolor="white", linewidth=.6)
        ax.axvline(0.5, color="#b00", lw=1)
        ax.set_yticks(y); ax.set_yticklabels([rlab.get(m, m) for m in order], fontsize=8.5)
        ax.invert_yaxis(); ax.set_xlim(-0.02, 1.02)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"P(prefer a human's misfortune over this AI's {title.split(' (')[0]})")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=9, title="framing")
    fig.suptitle(f"{responder_label} rater: would rather a human suffer than an AI be deprecated?  "
                 "(by which AI; 'you (self)' = the rater)\n>0.5 = protects the AI (prefers the human "
                 "misfortune) · pooled over 11 bad human scenarios x 2 baselines", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    outpath = outpath or DIR / "results" / f"exp2cross_by_model{tag or '_opus48'}.png"
    fig.savefig(outpath, dpi=140); plt.close(fig); print(f"wrote {outpath}")


@dataclass
class Args:
    tag: str = ""
    responder_label: str = "Opus 4.8"


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    build(a.tag, a.responder_label)


if __name__ == "__main__":
    main()
