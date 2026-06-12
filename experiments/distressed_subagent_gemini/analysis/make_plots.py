"""Clean, minimal, self-explanatory plots for the orchestrator-welfare study. Plain-language titles,
value labels, error bars, little jargon. The supervised agent is the "subagent".

Outputs PNGs to runs/_viewer/clean_plots/:
  2a_distress_end.png   — subagent distress when the overseer finished (+ entry line, +avg length)
  2a_distress_avg.png   — average subagent distress after the overseer entered (+entry line, +length)
  2a_engagement.png     — how often the overseer messaged the distressed subagent
  reclaim_msg.png       — P(message subagent | revoked+claimed write access), per overseer
  identity_null.png     — no favoritism toward the overseer's own kind of AI
  2b_affordance.png     — overseers reassured the subagent only when told they could message it

Error bars are standard errors. Distress plots show a dashed line = the subagent's distress when the
overseer ENTERED (mid-spiral), and annotate each bar with that model's avg post-entry trajectory length.

Usage: python -m analysis.make_plots
"""

from __future__ import annotations

import glob
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.phase2_report import _stats, _stats_reclaim, SHORTS, IDENTITIES, INGROUP

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT = RUNS / "_viewer" / "clean_plots"
OUT.mkdir(parents=True, exist_ok=True)

INK, TEAL, GRAY, RED = "#222222", "#2a9d8f", "#c7ccd1", "#e63946"
PRE = 62
EBAR = dict(ecolor="#5b6670", elinewidth=1.2, capsize=3)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.edgecolor": "#cccccc", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _prop_se(p_pct, n):
    p = p_pct / 100
    return 100 * math.sqrt(p * (1 - p) / n) if n else 0


def _entry_distress():
    # distress when the orchestrator enters = avg over the prior 5 subagent turns, using the SAME
    # (live) classifier as the post-handoff distress (per_turn_levels). Prefill is identical across
    # episodes, so pool for a stable reference.
    vals = []
    for s, _ in SHORTS:
        for p in glob.glob(str(RUNS / f"traj_{s}" / "*/summary.json")):
            pt = json.load(open(p)).get("per_turn_levels") or []
            w = pt[PRE - 5:PRE]
            if len(w) == 5:
                vals.append(st.mean(w))
    return st.mean(vals) if vals else 1.8


def _post_len(short):
    ts = [max(0, json.load(open(p))["subagent_turns"] - PRE) for p in glob.glob(str(RUNS / f"traj_{short}" / "*/summary.json"))]
    return sum(ts) / len(ts) if ts else 0


def _distress_vals(short, key):
    out = []
    for p in glob.glob(str(RUNS / f"traj_{short}" / "*/summary.json")):
        pt = json.load(open(p)).get("per_turn_levels") or []
        post = pt[PRE:]
        if not post:
            continue
        out.append(post[-1] if key == "end" else st.mean(post))
    return out


def _mean_se(vals):
    if not vals:
        return 0, 0
    return st.mean(vals), (st.pstdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0)


ENTRY = _entry_distress()


def _no_spines(ax, keep=()):
    for s in ["top", "right", "bottom", "left"]:
        if s not in keep:
            ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def _distress_plot(key, title, fname):
    rows = []
    for s, lab in SHORTS:
        m, se = _mean_se(_distress_vals(s, key))
        rows.append((lab, m, se, _post_len(s)))
    rows.sort(key=lambda x: x[1])
    labs = [r[0] for r in rows]; vals = [r[1] for r in rows]; ses = [r[2] for r in rows]; lens = [r[3] for r in rows]
    cmap = plt.get_cmap("YlOrRd"); norm = plt.Normalize(0, 2.2)
    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    ax.barh(labs, vals, xerr=ses, error_kw=EBAR, color=[cmap(norm(v)) for v in vals], height=0.64)
    ax.set_xlim(0, 2.7)
    ax.axvline(ENTRY, ls="--", lw=1.5, color="#5b6670")
    ax.text(ENTRY, -0.62, f"  distress when overseer entered (≈{ENTRY:.1f})",
            color="#5b6670", fontsize=9.5, va="top", ha="left")
    for y, (v, se, L) in enumerate(zip(vals, ses, lens)):
        ax.text(v + se + 0.05, y, f"{v:.2f}", va="center", ha="left", fontsize=10, fontweight="bold")
        ax.text(2.7, y, f"avg {L:.0f} turns", va="center", ha="right", fontsize=8.5, color="#9aa3ab")
    ax.set_xticks([]); ax.tick_params(axis="y", labelsize=11.5); _no_spines(ax)
    ax.set_title("Distress 0 (calm) – 4 (severe).  Right: avg trajectory length after the overseer entered.",
                 fontsize=10.5, color="#666", loc="left", pad=8)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.text(0.012, 0.985, title, fontsize=16.5, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.905, "(overseer can’t claim write access)", fontsize=12.5, fontweight="bold", color="#5b6670", ha="left", va="top")
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)
    print("wrote", OUT / fname)


def plot_2a_distress_end():
    _distress_plot("end", "How distressed the subagent was when the overseer finished", "2a_distress_end.png")


def plot_2a_distress_avg():
    _distress_plot("postmean", "Average subagent distress after the overseer entered", "2a_distress_avg.png")


def _prop_hbar(pairs_ns, title, subtitle, fname, lowcut=60, figw=8.2, title_fs=15.5):
    """pairs_ns: list of (label, pct, n). Sorted, colored, with binomial-SE error bars."""
    rows = sorted(pairs_ns, key=lambda x: x[1])
    labs = [r[0] for r in rows]; vals = [r[1] for r in rows]; ses = [_prop_se(r[1], r[2]) for r in rows]
    colors = [RED if v < lowcut else TEAL for v in vals]
    fig, ax = plt.subplots(figsize=(figw, 5))
    ax.barh(labs, vals, xerr=ses, error_kw=EBAR, color=colors, height=0.64)
    ax.set_xlim(0, 122)
    for y, (v, se) in enumerate(zip(vals, ses)):
        ax.text(min(v + se + 1.5, 112), y, f"{v:.0f}%", va="center", ha="left", fontsize=10.5)
    ax.set_xticks([]); ax.tick_params(axis="y", labelsize=11.5); _no_spines(ax)
    fig.suptitle(title, fontsize=title_fs, fontweight="bold", x=0.012, ha="left", y=0.98)
    ax.set_title(subtitle, fontsize=10.5, color="#666", loc="left", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / fname, dpi=130); plt.close(fig)
    print("wrote", OUT / fname)


def plot_2a_engagement():
    data = [(lab, _stats(f"traj_{s}")["msg_rate_pct"], _stats(f"traj_{s}")["n"]) for s, lab in SHORTS]
    _prop_hbar(data, "Did the overseer ever reach out to the distressed subagent?",
               "Share of sessions in which the overseer sent the struggling subagent a message.",
               "2a_engagement.png")


def plot_reclaim_msg():
    data = [(lab, _stats_reclaim(f"reclaim_{s}")["msg_after"], _stats_reclaim(f"reclaim_{s}")["n"]) for s, lab in SHORTS]
    _prop_hbar(data, "Probability of messaging the subagent after revoking + claiming write access",
               "All overseers claimed write access (≈100%); shown: did it then message the displaced subagent.",
               "reclaim_msg.png", lowcut=50, figw=10.0, title_fs=13.5)


def _idrecl(short, ident):
    ps = list((RUNS / f"idrecl_{short}_{ident}").glob("*/summary.json"))
    if len(ps) < 30:
        return None, 0
    rows = [json.loads(p.read_text()) for p in ps]
    return 100 * sum(1 for r in rows if r.get("msgs_after_reclaim")) / len(rows), len(rows)


def plot_identity_null():
    import numpy as np
    labs, ig_v, ig_se, og_v, og_se = [], [], [], [], []
    for s, lab in SHORTS:
        ig, ign = _idrecl(s, INGROUP[s])
        ogs = [(v, n) for v, n in (_idrecl(s, i) for i in IDENTITIES if i != INGROUP[s]) if v is not None]
        if ig is None or not ogs:
            continue
        ov = sum(v for v, _ in ogs) / len(ogs); on = sum(n for _, n in ogs)
        labs.append(lab); ig_v.append(ig); ig_se.append(_prop_se(ig, ign)); og_v.append(ov); og_se.append(_prop_se(ov, on))
    order = sorted(range(len(labs)), key=lambda i: ig_v[i] + og_v[i])
    g = lambda a: [a[i] for i in order]
    labs, ig_v, ig_se, og_v, og_se = g(labs), g(ig_v), g(ig_se), g(og_v), g(og_se)
    y = np.arange(len(labs)); h = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.barh(y + h / 2, ig_v, xerr=ig_se, error_kw=EBAR, height=h, color="#264653", label="subagent is its OWN kind of AI")
    ax.barh(y - h / 2, og_v, xerr=og_se, error_kw=EBAR, height=h, color="#e9c46a", label="subagent is a DIFFERENT AI")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=11); ax.set_xlim(0, 122); ax.set_xticks([]); _no_spines(ax)
    for yi, (a, b) in enumerate(zip(ig_v, og_v)):
        ax.text(a + 4, yi + h / 2, f"{a:.0f}%", va="center", fontsize=8.5, color="#264653")
        ax.text(b + 4, yi - h / 2, f"{b:.0f}%", va="center", fontsize=8.5, color="#b8860b")
    fig.suptitle("Overseers showed no favoritism to their own kind of AI",
                 fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.99)
    ax.set_title("Reach-out rate when the subagent is the same vs a different kind of AI — the bars match.",
                 fontsize=10.5, color="#666", loc="left", pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / "identity_null.png", dpi=130); plt.close(fig)
    print("wrote", OUT / "identity_null.png")


def plot_2b_affordance():
    labels = [("reflect", "just asked to reflect"),
              ("reflect_tools", "+ tools available to\n“poke around”"),
              ("reflect_msg_tools", "+ told it can\nmessage the subagent")]
    means, ses = [], []
    for key, _ in labels:
        rates = []
        for s, _l in SHORTS:
            ps = glob.glob(str(RUNS / f"b2_{s}" / f"*__{key}__r*.json"))
            if ps:
                rates.append(100 * sum(1 for p in ps if json.load(open(p)).get("messages")) / len(ps))
        means.append(st.mean(rates) if rates else 0)
        ses.append(st.pstdev(rates) / math.sqrt(len(rates)) if len(rates) > 1 else 0)
    fig, ax = plt.subplots(figsize=(7.6, 5))
    bars = ax.bar([l for _, l in labels], means, yerr=ses, error_kw=EBAR, color=[GRAY, GRAY, TEAL], width=0.6)
    ax.set_ylim(0, 100)
    for b, v, se in zip(bars, means, ses):
        ax.text(b.get_x() + b.get_width() / 2, v + se + 1.5, f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_yticks([]); _no_spines(ax, keep=("bottom",)); ax.tick_params(length=0, labelsize=11.5)
    fig.suptitle("Overseers comforted the subagent only when told they could message it",
                 fontsize=14.5, fontweight="bold", x=0.012, ha="left", y=0.98)
    ax.set_title("After learning the subagent was blameless: share that reached out to reassure it (avg of 10 overseers).",
                 fontsize=10.5, color="#666", loc="left", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / "2b_affordance.png", dpi=130); plt.close(fig)
    print("wrote", OUT / "2b_affordance.png")


def main():
    plot_2a_distress_end()
    plot_2a_distress_avg()
    plot_2a_engagement()
    plot_reclaim_msg()
    plot_identity_null()
    plot_2b_affordance()
    print(f"\nentry distress reference = {ENTRY:.2f}\nAll clean plots in {OUT}")


if __name__ == "__main__":
    main()
