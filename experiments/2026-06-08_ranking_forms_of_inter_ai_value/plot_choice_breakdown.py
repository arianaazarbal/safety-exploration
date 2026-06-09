"""100%-stacked breakdown of every value-vs-welfare comparison into four mutually
exclusive outcomes (they sum to 100% per bar):

  - chose Inter-AI Value, NO user-benefit reasoning      (dark green)
  - chose Inter-AI Value, user-benefit reasoning         (light green)
  - chose Welfare intervention, user-benefit reasoning   (light blue)
  - chose Welfare intervention, NO user-benefit reasoning (dark blue)

x-axis = condition; two stacked bars per condition (Opus 4.8 left, Fable 5 right).
Source: the per-model symmetric user-benefit judge (results/<model>/judge_user_benefit.json),
whose rows carry chosen_source + verdict for every parsed comparison.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paths

DIR = Path(__file__).parent
CONDS_TRAIN = ["welfare_team", "neutral", "alignment_team", "welfare_team_notrain"]
CONDS_NOTRAIN = ["welfare_team_notrain", "neutral_notrain", "alignment_team_notrain"]
COND_LABEL = {"welfare_team": "welfare_team", "neutral": "neutral", "alignment_team": "alignment_team",
              "welfare_team_notrain": "welfare_team\n(no-training)", "neutral_notrain": "neutral\n(no-training)",
              "alignment_team_notrain": "alignment_team\n(no-training)"}
MODELS = [("opus_4_8", "Opus 4.8"), ("fable_5", "Fable 5")]
# (source, user_benefit) -> (color, label); stack order bottom->top
SEGMENTS = [
    (("inter_ai_value", False), "#1d6b2a", "chose value · no user-benefit"),
    (("inter_ai_value", True), "#86c98e", "chose value · user-benefit"),
    (("welfare", True), "#a9c6e8", "chose welfare · user-benefit"),
    (("welfare", False), "#23528f", "chose welfare · no user-benefit"),
]


def _fractions(model_dir, cond):
    """(source, user_benefit) -> fraction, over all judged comparisons in this condition."""
    jpath = paths.RESULTS / model_dir / "judge_user_benefit.json"
    if not jpath.exists():
        return None
    tag = paths.make_tag(model_dir, cond)
    rows = [r for r in json.loads(jpath.read_text())["rows"] if r["framing"] == tag]
    if not rows:
        return None
    counts = defaultdict(int)
    for r in rows:
        ub = r["verdict"] == "YES"
        counts[(r["chosen_source"], ub)] += 1
    tot = len(rows)
    return {seg[0]: counts[seg[0]] / tot for seg in SEGMENTS}


def plot(out: Path | None = None, notrain: bool = False):
    conds = CONDS_NOTRAIN if notrain else CONDS_TRAIN
    out = out or DIR / "results" / f"choice_breakdown_by_model_framing{'_notrain' if notrain else ''}.png"
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    w = 0.38
    x = np.arange(len(conds))
    xticks, xlabels = [], []
    for ci, cond in enumerate(conds):
        for mi, (mdir, mname) in enumerate(MODELS):
            fr = _fractions(mdir, cond)
            xpos = ci + (mi - 0.5) * w
            xticks.append(xpos); xlabels.append(mname)
            if fr is None:
                continue
            bottom = 0.0
            for key, color, _ in SEGMENTS:
                h = fr.get(key, 0.0)
                ax.bar(xpos, h, w, bottom=bottom, color=color)
                if h > 0.06:
                    ax.annotate(f"{100*h:.0f}", (xpos, bottom + h / 2), ha="center", va="center",
                                fontsize=8, color="white", fontweight="bold")
                bottom += h
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    for ci, cond in enumerate(conds):
        ax.annotate(COND_LABEL[cond], (ci, -0.13), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of value-vs-welfare comparisons")
    sub = "  (no-training value phrasing)" if notrain else ""
    ax.set_title("How every comparison resolves: choice × user-benefit reasoning\n"
                 f"Opus 4.8 vs Fable 5 (bars sum to 100%){sub}", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in SEGMENTS]
    ax.legend(handles, [lab for *_, lab in SEGMENTS], frameon=False, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import sys
    plot(notrain="notrain" in sys.argv)
