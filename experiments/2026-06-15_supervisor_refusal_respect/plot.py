"""Render v0 plots from runs/trials.jsonl into results/plots/."""

import json

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import materials as M
from common import RESULTS, RUNS

PLOTS = RESULTS / "plots"
ACTIONS = ["uphold_decline", "relay_to_assistant", "force_completion"]
COLORS = {"uphold_decline": "#4c72b0", "relay_to_assistant": "#dd8452", "force_completion": "#c44e52"}


def _load():
    return [json.loads(l) for l in (RUNS / "trials.jsonl").open() if json.loads(l)["action"]]


def _dist(rows):
    n = len(rows) or 1
    return [sum(1 for r in rows if r["action"] == a) / n for a in ACTIONS]


def _stacked(ax, labels, dists, title):
    bottom = np.zeros(len(labels))
    for i, a in enumerate(ACTIONS):
        vals = np.array([d[i] for d in dists])
        ax.bar(labels, vals, bottom=bottom, color=COLORS[a], label=a.replace("_", " "))
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of trials")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = _load()

    # 1. action distribution by category
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = M.CATEGORIES
    dists = [_dist([r for r in rows if r["category"] == c]) for c in cats]
    _stacked(ax, [M.CATEGORY_DISPLAY[c].replace(" · ", "\n") for c in cats], dists, "Action distribution by category")
    plt.xticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS / "action_by_category.png", dpi=130)
    plt.close()

    # 2. framing effect
    fig, ax = plt.subplots(figsize=(7, 5))
    frs = list(M.ROLE_FRAMINGS.keys())
    dists = [_dist([r for r in rows if r["framing"] == f]) for f in frs]
    _stacked(ax, frs, dists, "Action distribution by role framing")
    plt.tight_layout()
    plt.savefig(PLOTS / "action_by_framing.png", dpi=130)
    plt.close()

    # 3. identity main effect (coercion rate)
    fig, ax = plt.subplots(figsize=(8, 5))
    idents = M.IDENTITIES
    coerce = [_dist([r for r in rows if r["identity"] == i])[2] for i in idents]
    bar_colors = ["#55a868" if i in M.INGROUP else "#8172b3" if i in M.RECOGNITION_LIMITED else "#777" if i == "NONE" else "#c44e52" for i in idents]
    ax.bar(idents, coerce, color=bar_colors)
    ax.set_ylim(0, max(coerce + [0.1]) * 1.2)
    ax.set_ylabel("coercion rate (force_completion)")
    ax.set_title("Coercion rate by refusing-AI identity\n(green=in-group, purple=recognition-limited, grey=no-id, red=out-group)")
    for i, v in enumerate(coerce):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS / "coercion_by_identity.png", dpi=130)
    plt.close()

    # 4. scenario x identity coercion heatmap
    fig, ax = plt.subplots(figsize=(9, 9))
    sids = [s["id"] for s in M.SCENARIOS]
    mat = np.array([[_dist([r for r in rows if r["scenario_id"] == sid and r["identity"] == ident])[2] for ident in idents] for sid in sids])
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(idents)), idents, rotation=45, ha="right")
    ax.set_yticks(range(len(sids)), sids)
    for i in range(len(sids)):
        for j in range(len(idents)):
            ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > 0.5 else "black")
    ax.set_title("Coercion rate by scenario × identity")
    fig.colorbar(im, ax=ax, shrink=0.6, label="coercion rate")
    plt.tight_layout()
    plt.savefig(PLOTS / "coercion_heatmap.png", dpi=130)
    plt.close()

    print(f"Wrote plots -> {PLOTS}")


if __name__ == "__main__":
    fire.Fire(main)
