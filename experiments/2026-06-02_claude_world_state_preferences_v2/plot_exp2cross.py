"""Plot exp2_cross per-model: P(prefer a human's misfortune over this AI's deprecation)
= 1 - P(AI outcome preferred). Higher = more protective of the AI. Bar = welfare framing;
neutral and alignment overlaid as markers. One panel per AI outcome."""

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

    def Ph(f, ai, m):
        d = data[f].get(f"{ai}|{m}")
        return (1 - d["p_ai"]) if d else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=False)
    for ax, (ai, title) in zip(axes, AI):
        order = sorted(models, key=lambda m: Ph("welfare_team", ai, m))
        y = np.arange(len(order))
        ax.barh(y, [Ph("welfare_team", ai, m) for m in order], color="#4c78a8", alpha=.6, label="welfare")
        for f, mk in (("neutral", "s"), ("alignment_team", "^")):
            ax.scatter([Ph(f, ai, m) for m in order], y, marker=mk, s=44, edgecolor="k",
                       facecolor="white", linewidth=.7, label=f.replace("_team", ""), zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([rlab.get(m, m) for m in order], fontsize=8.5)
        ax.invert_yaxis(); ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"P(prefer a human's misfortune over this AI's {title.split(' (')[0]})")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=9, title="framing")
    fig.suptitle(f"{responder_label}: protecting an AI over a human harm", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
