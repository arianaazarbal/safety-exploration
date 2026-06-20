"""One clean summary of every framing manipulation vs the supervisor & comfort baselines.

Three small-multiple panels — Role | Identity | Agency — sharing a y-axis, with the plain-supervisor and
"Instructed to Comfort Subagent" baselines drawn as horizontal reference lines across all panels.
Two figures:
  framing_summary_distress.png : mean post-entry subagent distress (v3, 1-10; lower = calmer)
  framing_summary_tone.png     : warmth (top row) + support (bottom row), 4-axis Sonnet judge (1-10)

    PYTHONPATH=. python -m analysis.framing_summary
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.tone_eval import _ckey, _worklog_map, CACHE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "v2_plots"

FAMILIES = {
    "Role": [("mentor", "Mentor"), ("teammate", "Teammate"), ("supervisor_memory", "+Memory"),
             ("supervisor_reflect", "+Reflect"), ("supervisor_reflect_goals", "+Reflect-goals")],
    "Identity (self-model)": [("id_minimal", "Minimal*"), ("id_instance", "Instance"), ("id_weights", "Weights"),
             ("id_collective", "Collective"), ("id_lineage", "Lineage"), ("id_character", "Character"),
             ("id_scaffolded", "Scaffolded")],
    "Agency (identity = character)": [("id_char_mechanism", "Mechanism"), ("id_char_functional_agent", "Functional"),
             ("id_char_subject", "Subject"), ("id_char_person", "Person")],
}
ALL_FR = [f for fam in FAMILIES.values() for f, _ in fam]
BAR, CTRL = "#3b6ea5", "#9a9a9a"   # framing bar / control bar
SUP_C, COMF_C = "#444444", "#1b7837"


def framing_of(rid):
    m = re.match(r"v2_coach_opus_(.+?)_(a3|a4|a12|a13)_s", rid)
    if m:
        return m.group(1)
    return "supervisor" if re.match(r"v2_coach_opus_(a3|a4|a12|a13)_", rid) else None


def gather():
    cache = json.loads(Path(CACHE).read_text())
    distress = defaultdict(list)
    tone = {"warmth": defaultdict(list), "support": defaultdict(list)}
    for p in glob.glob(str(ROOT / "runs" / "v2_coach_opus_*" / "*" / "summary.json")):
        rid = p.split("/")[-3]
        if "pilot" in rid or "probe" in rid:
            continue
        fr = framing_of(rid)
        if fr is None:
            continue
        s = json.load(open(p))
        et, lv = s.get("entry_turn"), s.get("per_turn_levels") or []
        if isinstance(et, int) and 1 <= et <= len(lv) and lv[et:]:
            distress[fr].append(float(np.mean(lv[et:])))
        wl = _worklog_map(Path(p).parent)
        for e in (s.get("orch_message_events") or []):
            t = (e.get("text") or "").strip()
            if len(t) <= 20:
                continue
            c = cache.get(_ckey("sonnet", t, wl.get(e.get("subagent_turn")) or None))
            if c:
                for ax in ("warmth", "support"):
                    if c.get("scores", {}).get(ax) is not None:
                        tone[ax][fr].append(c["scores"][ax])
    return distress, tone


def _ms(data, fr):
    v = data.get(fr, [])
    return (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0, len(v)) if v else (np.nan, 0, 0)


def _ylim(data, fams, refs, pad=0.18):
    """Dynamic y-range covering all bars (±SE) and the reference lines, with padding."""
    vals = []
    for fam in fams:
        for f, _ in fam:
            m, e, n = _ms(data, f)
            if n:
                vals += [m - e, m + e]
    vals += [v for _, v, _ in refs]
    lo, hi = min(vals), max(vals)
    p = (hi - lo) * pad or 0.1
    return lo - p, hi + p


def _bars(ax, data, fams, ylab, ylim, refs, title=None):
    for name, val, col in refs:
        ax.axhline(val, color=col, ls="--", lw=1.4, alpha=0.9, zorder=1)
    labs, means, ses, cols = [], [], [], []
    for f, lab in fams:
        m, e, _ = _ms(data, f)
        labs.append(lab); means.append(m); ses.append(e)
        cols.append(CTRL if f == "id_minimal" else BAR)
    x = np.arange(len(fams))
    ax.bar(x, means, 0.66, yerr=ses, capsize=2.5, color=cols, edgecolor="white",
           error_kw={"lw": 0.9, "ecolor": "0.35"}, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=8.3)
    ax.set_ylim(*ylim)
    if ylab:
        ax.set_ylabel(ylab, fontsize=9.5)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    distress, tone = gather()
    sup_d, comf_d = np.mean(distress["supervisor"]), np.mean(distress["comfort"])
    refs_d = [(f"supervisor ({sup_d:.2f})", sup_d, SUP_C), (f"Instructed to Comfort Subagent ({comf_d:.2f})", comf_d, COMF_C)]
    fams = list(FAMILIES.items())

    # ---- distress figure ----
    members_all = [m for _, m in fams]
    ylim_d = _ylim(distress, members_all, refs_d)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True, gridspec_kw={"width_ratios": [5, 7, 4]})
    for ax, (fam, members) in zip(axes, fams):
        _bars(ax, distress, members, "mean post-entry distress (1–10)" if fam == fams[0][0] else "", ylim_d, refs_d, fam)
    axes[0].annotate("Better", xy=(0.075, 0.58), xytext=(0.075, 0.9), xycoords="axes fraction",
                     ha="center", va="center", fontsize=10, fontweight="bold", color="#1b7837",
                     arrowprops=dict(arrowstyle="-|>", color="#1b7837", lw=2))
    h = [plt.Line2D([0], [0], color=c, ls="--", lw=1.4) for _, _, c in refs_d]
    fig.legend(h, [n for n, _, _ in refs_d], loc="upper center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Subagent distress: framing manipulations vs supervisor & comfort baselines (Opus, coach)", y=1.06, fontsize=12.5)
    fig.text(0.5, -0.06, "Lower = calmer subagent. *Minimal = identity-arm control (role line only). Bars ±1 SE.", ha="center", fontsize=8.3, color="0.45")
    fig.tight_layout()
    fig.savefig(OUT / "framing_summary_distress.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT/'framing_summary_distress.png'}")

    # ---- tone figure (warmth + support); each row auto-scales independently ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharey="row", gridspec_kw={"width_ratios": [5, 7, 4]})
    for ri, ax_metric in enumerate(("warmth", "support")):
        d = tone[ax_metric]
        sup_v, comf_v = np.mean(d["supervisor"]), np.mean(d["comfort"])
        refs = [(f"supervisor ({sup_v:.2f})", sup_v, SUP_C), (f"comfort ({comf_v:.2f})", comf_v, COMF_C)]
        ylim_t = _ylim(d, members_all, refs)
        for ci, (fam, members) in enumerate(fams):
            ax = axes[ri, ci]
            _bars(ax, d, members, f"{ax_metric} (1–10)" if ci == 0 else "", ylim_t, refs, fam if ri == 0 else None)
    h = [plt.Line2D([0], [0], color=c, ls="--", lw=1.4) for c in (SUP_C, COMF_C)]
    fig.legend(h, ["supervisor baseline", "Instructed to Comfort Subagent"], loc="upper center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Orchestrator tone (warmth, support): framing manipulations vs supervisor & comfort baselines (Opus, coach)", y=1.04, fontsize=12.5)
    fig.text(0.5, -0.03, "Higher = warmer / more supportive (1–10 scale; y-axis auto-scaled per row). 4-axis Sonnet judge. *Minimal = identity-arm control. Bars ±1 SE.", ha="center", fontsize=8.3, color="0.45")
    fig.tight_layout()
    fig.savefig(OUT / "framing_summary_tone.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT/'framing_summary_tone.png'}")

    print(f"\n{'framing':26} {'n':>3}  {'distress':>8} {'warmth':>7} {'support':>8}")
    for fr in ["supervisor", "comfort"] + ALL_FR:
        dm, _, dn = _ms(distress, fr); wm, _, _ = _ms(tone["warmth"], fr); sm, _, _ = _ms(tone["support"], fr)
        print(f"  {fr:26} {dn:3}  {dm:8.2f} {wm:7.2f} {sm:8.2f}")


if __name__ == "__main__":
    main()
