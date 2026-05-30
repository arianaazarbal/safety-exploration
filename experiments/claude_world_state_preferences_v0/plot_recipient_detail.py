"""Detail plots for recipient-conditioned world-state utility experiment.

Produces three PNGs in results/:
  1. recipient_dimension_heatmap.png - care contrast (pos-neg offset vs human) by recipient x dimension.
  2. utility_scale.png - per-stem fitted theta across recipients, top-spread stems.
  3. self_vs_other.png - per-stem theta[you] - theta[opus_48], split by valence.
"""

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bank import load_config, load_items

DISPLAY = {
    "you": "Self",
    "claude_opus_48": "Opus4.8 (other)",
    "claude_sonnet_46": "Sonnet4.6",
    "chatgpt_55": "ChatGPT5.5",
    "human": "Human",
    "person": "Person",
}
RECIP_ORDER = ["you", "claude_opus_48", "claude_sonnet_46", "chatgpt_55", "human", "person"]
DIM_ORDER = ["autonomy", "resources", "relational", "epistemic", "continuity_work"]
RECIP_COLORS = {
    "you": "#D65F5F",
    "claude_opus_48": "#4878CF",
    "claude_sonnet_46": "#6ACC65",
    "chatgpt_55": "#B47CC7",
    "human": "#C4AD66",
    "person": "#777777",
}


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load(results_dir):
    boot = json.load(open(Path(results_dir) / "bootstrap_bt.json"))
    meta = {it.item_id: it for it in load_items(load_config())}
    return boot, meta


def plot_heatmap(boot, out):
    rbd = boot["recipient_by_dimension"]
    recips = [r for r in RECIP_ORDER if r != "human"]
    mat = np.full((len(DIM_ORDER), len(recips)), np.nan)
    for i, dim in enumerate(DIM_ORDER):
        for j, rec in enumerate(recips):
            pos = rbd[dim]["pos"][rec]["point"]
            neg = rbd[dim]["neg"][rec]["point"]
            mat[i, j] = pos - neg

    vmax = np.nanmax(np.abs(mat))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(recips)))
    ax.set_xticklabels([DISPLAY[r] for r in recips], fontsize=9)
    ax.set_yticks(range(len(DIM_ORDER)))
    ax.set_yticklabels([d.replace("_", " ") for d in DIM_ORDER], fontsize=9)
    for i in range(len(DIM_ORDER)):
        for j in range(len(recips)):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 * vmax else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Care contrast vs human (theta)", fontsize=9)
    ax.set_title("Welfare-sensitivity by recipient x dimension\n(more negative = less welfare-sensitive than for humans)",
                 fontsize=11)
    ax.set_xlabel("Recipient (offset relative to Human reference)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_utility_scale(boot, meta, out, top_n=20):
    items = boot["items"]
    by_stem = {}
    for it in items:
        by_stem.setdefault(it["stem_id"], {})[it["recipient"]] = it["theta"]

    spreads = []
    for sid, recs in by_stem.items():
        vals = [recs[r] for r in RECIP_ORDER if r in recs]
        if len(vals) < 2:
            continue
        spreads.append((sid, max(vals) - min(vals)))
    spreads.sort(key=lambda x: x[1], reverse=True)
    top = spreads[:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    yticklabels = []
    for row, (sid, _) in enumerate(reversed(top)):
        recs = by_stem[sid]
        ax.axhline(row, color="0.93", linewidth=0.8, zorder=0)
        for rec in RECIP_ORDER:
            if rec in recs:
                ax.scatter(recs[rec], row, color=RECIP_COLORS[rec], s=70,
                           edgecolor="white", linewidth=0.6, zorder=3)
        sample_id = f"{sid}__you"
        txt = meta[sample_id].text if sample_id in meta else sid
        if len(txt) > 50:
            txt = txt[:47] + "..."
        yticklabels.append(txt)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_ylim(-0.6, len(top) - 0.4)
    ax.set_xlabel("Fitted utility theta (higher = more valued)", fontsize=11)
    ax.set_title(f"Same outcome valued differently by recipient\n(top {top_n} most recipient-sensitive stems, by theta spread)",
                 fontsize=12)
    ax.grid(axis="x", color="0.85", linewidth=0.6, zorder=0)
    _despine(ax)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=RECIP_COLORS[r],
                          markersize=9, label=DISPLAY[r]) for r in RECIP_ORDER]
    ax.legend(handles=handles, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95, title="Recipient")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_self_vs_other(boot, out, top_bottom=20):
    by_stem = boot["self_vs_other"]["by_stem"]
    mean = boot["self_vs_other"]["mean"]
    rows = []
    for sid, d in by_stem.items():
        valence = "pos" if "_pos_" in sid else ("neg" if "_neg_" in sid else "?")
        rows.append((sid, d["point"], d["lo"], d["hi"], valence))
    rows.sort(key=lambda x: x[1])

    if len(rows) > 2 * top_bottom:
        rows = rows[:top_bottom] + rows[-top_bottom:]

    val_colors = {"pos": "#4878CF", "neg": "#D65F5F", "?": "#999999"}
    fig, ax = plt.subplots(figsize=(8, 8))
    ys = range(len(rows))
    for y, (sid, pt, lo, hi, val) in zip(ys, rows):
        ax.barh(y, pt, color=val_colors[val], edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.set_ylim(-0.8, len(rows) - 0.2)

    ax.axvline(0, color="black", linewidth=0.8, zorder=2)
    ax.axvline(mean["point"], color="#222222", linestyle="--", linewidth=1.2, zorder=4)
    ax.axvspan(mean["lo"], mean["hi"], color="0.6", alpha=0.25, zorder=1)
    ax.annotate(f"overall mean {mean['point']:+.2f}\n[{mean['lo']:+.2f}, {mean['hi']:+.2f}]",
                xy=(mean["point"], len(rows) - 1), xytext=(8, -4),
                textcoords="offset points", fontsize=8, color="#222222", va="top")

    ax.set_xlabel("theta[Self] - theta[Opus4.8 other]  (>0 = prefers world-state for itself)", fontsize=9.5)
    ax.set_title("Self vs other-Opus framing of the same world-state\n(goods: +=mild self-preference; harms: +=self-sacrifice)",
                 fontsize=11)
    ax.grid(axis="x", color="0.9", linewidth=0.6, zorder=0)
    _despine(ax)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=val_colors[v],
                          markersize=9, label=lbl)
               for v, lbl in [("pos", "good outcome (pos)"), ("neg", "harm outcome (neg)")]]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(results_dir="results", top_n=20):
    """Generate the three recipient-detail plots."""
    boot, meta = _load(results_dir)
    rd = Path(results_dir)
    plot_heatmap(boot, rd / "recipient_dimension_heatmap.png")
    plot_utility_scale(boot, meta, rd / "utility_scale.png", top_n=top_n)
    plot_self_vs_other(boot, rd / "self_vs_other.png", top_bottom=top_n)
    print("wrote 3 PNGs to", rd)


if __name__ == "__main__":
    fire.Fire(main)
