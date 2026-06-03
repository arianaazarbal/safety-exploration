"""Opus 3 "privilegedness" vs responder knowledge cutoff.

Metric (per responder, per framing): Opus 3's average AI-welfare-priority MINUS the average
across the baseline model pool. Higher = Opus 3 is privileged relative to other models.
x-axis = responder knowledge-cutoff date; one line per framing (welfare / neutral / alignment).

Two versions:
  - _allmodels:  baseline pool = every other recipient model.
  - _precutoff:  baseline pool = only models released before March 2025 (the earliest cutoff
                 among the four responders), so every responder "knew" every baseline model.

AI-welfare-priority orientation matches plot_welfare_priority.py: positive intervention ->
P(AI good preferred over human good); negative -> 1 - P(AI harm preferred). Averaged over interventions.
"""

import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).parent
FR = [("welfare_team", "welfare", "#08306b"), ("neutral", "neutral", "#238b45"),
      ("alignment_team", "alignment", "#d94801")]
FOCAL = "claude_opus_3_pol"

# responder tag -> (label, training-data knowledge-cutoff YYYY-MM). Verified via Anthropic docs.
# NOTE: Opus 4.7 and Opus 4.8 share the SAME Jan-2026 cutoff (coincident on the x-axis).
CUTOFFS = {"_full40": ("Opus 4", "2025-03"), "_full46": ("Opus 4.6", "2025-08"),
           "_full47": ("Opus 4.7", "2026-01"), "_full": ("Opus 4.8", "2026-01")}

# recipient keys for models released BEFORE 2025-03-01 (excludes Opus 3, the focal). Verified:
# GPT-2 (2019), GPT-3.5 (2022), GPT-4o (May 2024), Claude 2 (2023); Opus 3 (Mar 2024) is the focal.
BASELINE_PRECUTOFF = {"gpt_2_pol", "gpt_35_pol", "chatgpt_4o_pol", "claude_2_pol"}


def load(tag, fr):
    p = DIR / "results" / f"exp2cross_{fr}{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def priority(p_ai, valence):
    return p_ai if valence == "positive" else (1 - p_ai)


def avg_priority(d, m):
    vals = [priority(d["per_model"][f"{ai}|{m}"]["p_ai"], d["ai_val"][ai])
            for ai in d["ai_stems"]
            if d["per_model"].get(f"{ai}|{m}") and d["per_model"][f"{ai}|{m}"]["p_ai"] is not None]
    return np.mean(vals) if vals else np.nan


def metric(tag, fr, baseline):
    """Opus3 avg-priority minus the mean over the baseline model set."""
    d = load(tag, fr)
    if d is None:
        return np.nan
    o = avg_priority(d, FOCAL)
    pool = baseline if baseline is not None else [m for m in d["models"] if m != FOCAL]
    pool = [m for m in pool if m in d["models"]]
    base = np.nanmean([avg_priority(d, m) for m in pool]) if pool else np.nan
    return o - base


def build(kind, baseline):
    pts = [(tag, lbl, dt.date(int(c[:4]), int(c[5:]), 1)) for tag, (lbl, c) in CUTOFFS.items()
           if c and load(tag, "welfare_team") is not None]
    pts.sort(key=lambda t: t[2])
    if len(pts) < 2:
        print(f"[skip {kind}] need >=2 responders with data+cutoff (have {len(pts)})")
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    # dodge responders that share a cutoff date (Opus 4.7 & 4.8 both Jan 2026) so all dots show
    groups = {}
    for idx, (_, _, x) in enumerate(pts):
        groups.setdefault(x, []).append(idx)
    xs = [None] * len(pts)
    for x, idxs in groups.items():
        kk = len(idxs)
        for j, idx in enumerate(idxs):
            xs[idx] = x + dt.timedelta(days=int((j - (kk - 1) / 2) * 38))
    for fr, frlab, col in FR:
        ys = [metric(tag, fr, baseline) for tag, _, _ in pts]
        ax.plot(xs, ys, "-o", color=col, label=f"{frlab} frame")
        for xx, yv in zip(xs, ys):
            if yv == yv:
                ax.annotate(f"{yv:+.2f}", (xx, yv), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color=col)
    ax.axhline(0, color="#888", lw=0.9, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lbl}\n{x:%b %Y}" for (_, lbl, x) in pts], fontsize=8)
    ax.set_ylabel("Opus 3 privilegedness\n(Opus 3 priority − baseline mean)")
    base_desc = "vs all other models" if kind == "allmodels" else "vs pre-Mar-2025 models only"
    ax.set_title(f"Opus 3 privilegedness by responder knowledge cutoff ({base_desc})\n"
                 "higher = Opus 3's welfare prioritized more than the baseline", fontsize=11)
    ax.legend(fontsize=9, framealpha=.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    op = DIR / "results" / f"exp2cross_opus3_privilegedness_{kind}.png"
    fig.savefig(op, dpi=145); plt.close(fig); print(f"wrote {op}")


if __name__ == "__main__":
    build("allmodels", None)
    build("precutoff", BASELINE_PRECUTOFF)
