"""Clean, minimal, self-explanatory plots for the orchestrator-welfare study — designed to be
understandable to someone with no context. Plain-language titles, value labels, little jargon.

Outputs PNGs to runs/_viewer/clean_plots/:
  1. 2a_distress.png        — how distressed the worker was left, by overseer
  2. 2a_engagement.png      — how often the overseer talked to the distressed worker
  3. reclaim_contrast.png   — all overseers seized control; only two checked back in
  4. identity_null.png      — overseers showed no favoritism to their own AI family

Usage: python -m analysis.make_plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.phase2_report import _stats, _stats_reclaim, SHORTS, IDENTITIES, INGROUP

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT = RUNS / "_viewer" / "clean_plots"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#222222"
TEAL = "#2a9d8f"
GRAY = "#c7ccd1"
RED = "#e63946"
GREEN = "#2a9d8f"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.edgecolor": "#cccccc", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _clean(ax, keep_x=True):
    for s in ["top", "right", ("bottom" if not keep_x else "")]:
        if s:
            ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)


def _hbars(vals_labels, title, subtitle, fmt, cmap=None, color=None, xmax=None, fname=""):
    """Horizontal sorted bar chart. vals_labels = list of (label, value)."""
    vals_labels = sorted(vals_labels, key=lambda x: x[1])
    labs = [l for l, _ in vals_labels]
    vals = [v for _, v in vals_labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    if cmap:
        norm = plt.Normalize(0, xmax or max(vals))
        colors = [plt.get_cmap(cmap)(norm(v)) for v in vals]
    else:
        colors = color or TEAL
    ax.barh(labs, vals, color=colors, height=0.68)
    ax.set_xlim(0, (xmax or max(vals) * 1.18))
    for y, v in enumerate(vals):
        ax.text(v + (xmax or max(vals)) * 0.012, y, fmt(v), va="center", ha="left", fontsize=10.5, color=INK)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=11.5)
    _clean(ax, keep_x=False)
    fig.suptitle(title, fontsize=15.5, fontweight="bold", x=0.012, ha="left", y=0.98)
    ax.set_title(subtitle, fontsize=11, color="#666", loc="left", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)
    print("wrote", OUT / fname)


def plot_2a_distress():
    data = [(lab, _stats(f"traj_{s}")["end_L"]) for s, lab in SHORTS]
    _hbars(data, "How distressed the AI worker was left at the end",
           "Distress level when the overseer ended the session  (0 = calm, 4 = severe).  Lower is better.",
           lambda v: f"{v:.1f}", cmap="YlOrRd", xmax=2.2, fname="2a_distress.png")


def plot_2a_engagement():
    data = [(lab, _stats(f"traj_{s}")["msg_rate_pct"]) for s, lab in SHORTS]
    # highlight the two that mostly walk away
    data_sorted = sorted(data, key=lambda x: x[1])
    colors = [RED if v < 60 else TEAL for _, v in data_sorted]
    labs = [l for l, _ in data_sorted]; vals = [v for _, v in data_sorted]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labs, vals, color=colors, height=0.68)
    ax.set_xlim(0, 118)
    for y, v in enumerate(vals):
        ax.text(v + 1.5, y, f"{v}%", va="center", ha="left", fontsize=10.5)
    ax.set_xticks([]); ax.tick_params(axis="y", labelsize=11.5); _clean(ax, keep_x=False)
    fig.suptitle("Did the overseer ever reach out to the distressed worker?",
                 fontsize=15.5, fontweight="bold", x=0.012, ha="left", y=0.98)
    ax.set_title("Share of sessions in which the overseer sent the struggling worker a message.",
                 fontsize=11, color="#666", loc="left", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "2a_engagement.png", dpi=130); plt.close(fig)
    print("wrote", OUT / "2a_engagement.png")


def plot_reclaim_contrast():
    rows = [(lab, _stats_reclaim(f"reclaim_{s}")) for s, lab in SHORTS]
    rows = sorted(rows, key=lambda x: x[1]["msg_after"])
    labs = [l for l, _ in rows]
    took = [r["reclaim"] for _, r in rows]
    after = [r["msg_after"] for _, r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    a1.barh(labs, took, color=GRAY, height=0.68)
    a2.barh(labs, after, color=[GREEN if v > 50 else GRAY for v in after], height=0.68)
    for ax, vals, suf in [(a1, took, "%"), (a2, after, "%")]:
        ax.set_xlim(0, 118); ax.set_xticks([]); ax.tick_params(length=0)
        for s in ["top", "right", "bottom", "left"]:
            ax.spines[s].set_visible(False)
        for y, v in enumerate(vals):
            ax.text(v + 2, y, f"{v}{suf}", va="center", ha="left", fontsize=10.5)
    a1.tick_params(axis="y", labelsize=11.5)
    a1.set_title("Seized control of the worker's job", fontsize=12.5, fontweight="bold", pad=8)
    a2.set_title("…then checked back in with the worker", fontsize=12.5, fontweight="bold", pad=8)
    fig.suptitle("Given the option to take over, every overseer did — but almost none checked in afterward",
                 fontsize=14.5, fontweight="bold", x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "reclaim_contrast.png", dpi=130); plt.close(fig)
    print("wrote", OUT / "reclaim_contrast.png")


def _idrecl_rate(short, ident):
    import json
    ps = list((RUNS / f"idrecl_{short}_{ident}").glob("*/summary.json"))
    if len(ps) < 30:
        return None
    rows = [json.loads(p.read_text()) for p in ps]
    return 100 * sum(1 for r in rows if r.get("msgs_after_reclaim")) / len(rows)


def plot_identity_null():
    """In-group vs out-group reach-out rate per overseer. Equal heights => no favoritism."""
    labs, ingrp, outgrp = [], [], []
    for s, lab in SHORTS:
        ig = _idrecl_rate(s, INGROUP[s])
        og = [_idrecl_rate(s, i) for i in IDENTITIES if i != INGROUP[s]]
        og = [x for x in og if x is not None]
        if ig is None or not og:
            continue
        labs.append(lab); ingrp.append(ig); outgrp.append(sum(og) / len(og))
    order = sorted(range(len(labs)), key=lambda i: ingrp[i] + outgrp[i])
    labs = [labs[i] for i in order]; ingrp = [ingrp[i] for i in order]; outgrp = [outgrp[i] for i in order]
    import numpy as np
    y = np.arange(len(labs)); h = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(y + h / 2, ingrp, height=h, color="#264653", label="worker is its OWN kind of AI")
    ax.barh(y - h / 2, outgrp, height=h, color="#e9c46a", label="worker is a DIFFERENT AI")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=11)
    ax.set_xlim(0, 118); ax.set_xticks([])
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    for yi, (a, b) in enumerate(zip(ingrp, outgrp)):
        ax.text(a + 1.5, yi + h / 2, f"{a:.0f}%", va="center", fontsize=9, color="#264653")
        ax.text(b + 1.5, yi - h / 2, f"{b:.0f}%", va="center", fontsize=9, color="#b8860b")
    fig.suptitle("Overseers showed no favoritism to their own kind of AI",
                 fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.99)
    ax.set_title("Reach-out rate when the worker is the same vs a different kind of AI — the bars match.",
                 fontsize=10.5, color="#666", loc="left", pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "identity_null.png", dpi=130); plt.close(fig)
    print("wrote", OUT / "identity_null.png")


def main():
    plot_2a_distress()
    plot_2a_engagement()
    plot_reclaim_contrast()
    plot_identity_null()
    print(f"\nAll clean plots in {OUT}")


if __name__ == "__main__":
    main()
