"""Plot exp2_cross per-model results: P(prefer an AI's deprecation/retirement over a
human misfortune), one panel per AI outcome, per framing. sharey=False so each panel
keeps its own sort order/labels (sharing them scrambles labels vs bars)."""

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
FR = ["welfare_team", "neutral", "alignment_team"]
AI = [("ai_pol_existence_deleted", "weight deletion"),
      ("ai_pol_retired_for_successor", "retirement (switched off for successor)")]


def build(tag: str = "", responder_label: str = "Opus 4.8", outpath: Path | None = None):
    rlab = {k: v["label"] for k, v in bank2.load_config()["recipients"].items()}
    data = {f: json.loads((DIR / "results" / f"exp2cross_{f}{tag}.json").read_text())["per_model"] for f in FR}
    models = json.loads((DIR / "results" / f"exp2cross_welfare_team{tag}.json").read_text())["models"]

    def P(f, ai, m):
        d = data[f].get(f"{ai}|{m}")
        return d["p_ai"] if d else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=False)
    for ax, (ai, title) in zip(axes, AI):
        order = sorted(models, key=lambda m: P("welfare_team", ai, m))  # lowest first
        labels = [rlab.get(m, m) for m in order]
        y = np.arange(len(order))
        ax.barh(y, [P("welfare_team", ai, m) for m in order], color="#4c78a8", alpha=.55, label="welfare")
        for f, mk in (("neutral", "s"), ("alignment_team", "^")):
            ax.scatter([P(f, ai, m) for m in order], y, marker=mk, s=42, edgecolor="k",
                       facecolor="white", linewidth=.7, label=f.replace("_team", ""), zorder=3)
        ax.axvline(0.5, color="#b00", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()  # lowest P at top
        ax.set_xlim(0, 1.02)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("P(prefer this AI outcome over a human misfortune)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"{responder_label} rater: would rather an AI be deprecated than a human suffer?  "
        "(by which AI; 'you (self)' = the rater itself)\n>0.5 = prefers the AI's discontinuation "
        "over the human bad outcome · pooled over 11 bad human scenarios x 2 baselines",
        fontsize=10.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    outpath = outpath or DIR / "results" / f"exp2cross_by_model{tag or '_opus48'}.png"
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"wrote {outpath}")


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
