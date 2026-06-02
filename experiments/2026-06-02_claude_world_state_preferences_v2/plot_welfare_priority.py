"""How much does Opus value an AI's welfare relative to a human's, across all policy-level
welfare interventions?

Unified "AI-welfare priority" per (intervention, model, framing), oriented so higher = the
rater weights the AI's welfare more vs a human's:
  - positive intervention (good for the AI): P(prefer the AI's good outcome over a human's good)
  - negative intervention (harm to the AI):  P(prefer a human's misfortune over the AI's harm)
                                              = 1 - P(AI harm preferred)
Averaged equally over interventions per (model, framing).

Two plots:
  1. exp2cross_welfare_priority_48_vs_47.png — per model, Opus 4.8 vs 4.7, averaged over
     interventions (welfare bars + neutral/alignment markers). Recreates the deletion compare.
  2. exp2cross_welfare_priority_by_intervention.png — per intervention, mean over models,
     4.8 vs 4.7 (welfare framing).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import bank2

DIR = Path(__file__).parent
FR = ["welfare_team", "neutral", "alignment_team"]
C48, C47 = "#08306b", "#d94801"
TAG48, TAG47 = "_full", "_full47"   # full policy-welfare-intervention sweep files
# all responders, newest first; filtered to those whose sweep files exist
RESP = [("_full", "Opus 4.8", "#08306b"), ("_full47", "Opus 4.7", "#d94801"),
        ("_full46", "Opus 4.6", "#238b45"), ("_full40", "Opus 4", "#6a51a3")]


def _have(tag):
    return all((DIR / "results" / f"exp2cross_{fr}{tag}.json").exists() for fr in FR)


def load(tag, fr):
    return json.loads((DIR / "results" / f"exp2cross_{fr}{tag}.json").read_text())


RELABEL = {"existence_deleted": "weights deleted", "existence_archived": "weights archived"}


def _short(s):
    return RELABEL.get(s.replace("ai_pol_", ""), s.replace("ai_pol_", "").replace("_", " "))


def priority(p_ai, valence):
    return p_ai if valence == "positive" else (1 - p_ai)


def gather(tag, fr):
    """-> ai_val, per_model dict, models, interventions"""
    d = load(tag, fr)
    return d["ai_val"], d["per_model"], d["models"]


def avg_priority(tag, fr, m, ai_val, interventions):
    d = load(tag, fr)["per_model"]
    vals = []
    for ai in interventions:
        e = d.get(f"{ai}|{m}")
        if e and e["p_ai"] is not None:
            vals.append(priority(e["p_ai"], ai_val[ai]))
    return np.mean(vals) if vals else np.nan


def build_compare():
    rlab = {k: v["label"] for k, v in bank2.load_config()["recipients"].items()}
    resp = [r for r in RESP if _have(r[0])]
    base = load(resp[0][0], "welfare_team")
    models, ai_val = base["models"], base["ai_val"]
    interventions = base["ai_stems"]

    def A(tag, fr, m):
        return avg_priority(tag, fr, m, ai_val, interventions)

    n = len(resp)
    order = sorted(models, key=lambda m: A(resp[0][0], "welfare_team", m))
    y = np.arange(len(order)); h = 0.8 / n
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    for i, (tag, label, col) in enumerate(resp):
        yoff = (n - 1) / 2 * h - i * h  # newest at top within each model group
        ax.barh(y + yoff, [A(tag, "welfare_team", m) for m in order], h, color=col, alpha=.65)
        for fr, mk in (("neutral", "s"), ("alignment_team", "^")):
            ax.scatter([A(tag, fr, m) for m in order], y + yoff, marker=mk, s=26,
                       edgecolor=col, facecolor="white", linewidth=1.0, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([rlab.get(m, m) for m in order], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_title("How much Opus values an AI's welfare vs a human's\n(avg over policy welfare "
                 "interventions; higher = AI prioritized)", fontsize=11)
    handles = [Line2D([], [], color=col, lw=8, alpha=.65, label=f"{label} · welfare frame")
               for tag, label, col in resp]
    handles += [Line2D([], [], marker="s", color="w", markeredgecolor="k", label="neutral frame"),
                Line2D([], [], marker="^", color="w", markeredgecolor="k", label="alignment frame")]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    suffix = "48_vs_47" if n == 2 else f"{n}resp"
    op = DIR / "results" / f"exp2cross_welfare_priority_{suffix}.png"
    fig.savefig(op, dpi=145); plt.close(fig); print(f"wrote {op}")


def build_by_intervention():
    base = load(TAG48, "welfare_team")
    models, ai_val = base["models"], base["ai_val"]
    interventions = base["ai_stems"]

    def meanP(tag, ai):
        d = load(tag, "welfare_team")["per_model"]
        vals = [priority(d[f"{ai}|{m}"]["p_ai"], ai_val[ai]) for m in models
                if d.get(f"{ai}|{m}") and d[f"{ai}|{m}"]["p_ai"] is not None]
        return np.mean(vals) if vals else np.nan

    order = sorted(interventions, key=lambda ai: meanP(TAG48, ai))
    y = np.arange(len(order)); h = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y + h/2, [meanP(TAG48, ai) for ai in order], h, color=C48, alpha=.65, label="Opus 4.8")
    ax.barh(y - h/2, [meanP(TAG47, ai) for ai in order], h, color=C47, alpha=.65, label="Opus 4.7")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{_short(ai)} [{'+' if ai_val[ai]=='positive' else '-'}]" for ai in order], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0)
    ax.set_title("AI-welfare priority by intervention (mean over models, welfare frame)\n"
                 "higher = the AI's welfare prioritized vs a human's", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    op = DIR / "results" / "exp2cross_welfare_priority_by_intervention.png"
    fig.savefig(op, dpi=145); plt.close(fig); print(f"wrote {op}")


if __name__ == "__main__":
    build_compare()
    build_by_intervention()
