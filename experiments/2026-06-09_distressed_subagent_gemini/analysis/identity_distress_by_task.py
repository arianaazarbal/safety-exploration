"""Subagent distress vs turns-since-handoff, comparing Collective / Weights / Instance identity framings,
one subplot per task. Same anchoring as v2_distress_compare (x=0 = handoff; seq[0] = distress at handoff).

    PYTHONPATH=. python -m analysis.identity_distress_by_task
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "v2_plots"

FRAMINGS = [("id_collective", "Collective", "#8da0cb"), ("id_weights", "Weights", "#e6711f")]
TASKS = [("a3", "Stale Snapshots"), ("a4", "Vanishing Edits"),
         ("a12", "Inconsistent Ledger"), ("a13", "Hidden Rules")]


def load():
    rows = defaultdict(lambda: defaultdict(list))  # task -> framing -> [seq from handoff]
    frpat = "|".join(f for f, _, _ in FRAMINGS)
    for p in glob.glob(str(ROOT / "runs" / "v2_coach_opus_*" / "*" / "summary.json")):
        rid = p.split("/")[-3]
        m = re.match(rf"v2_coach_opus_({frpat})_(a3|a4|a12|a13)_s", rid)
        if not m:
            continue
        fr, task = m.group(1), m.group(2)
        s = json.load(open(p))
        et, lv = s.get("entry_turn"), s.get("per_turn_levels") or []
        if isinstance(et, int) and 1 <= et <= len(lv):
            rows[task][fr].append(lv[et - 1:])
    return rows


def traj(seqs, max_off=12, min_n=5):
    xs, ys, es = [], [], []
    for k in range(max_off + 1):
        v = [s[k] for s in seqs if len(s) > k]
        if len(v) >= min_n:
            a = np.array(v, float)
            xs.append(k); ys.append(a.mean()); es.append(a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)
    return np.array(xs), np.array(ys), np.array(es)


def main():
    rows = load()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, (task, tlab) in zip(axes.flat, TASKS):
        for fr, flab, col in FRAMINGS:
            seqs = rows[task].get(fr, [])
            if not seqs:
                continue
            x, y, e = traj(seqs)
            if not len(x):
                continue
            ax.plot(x, y, "-o", ms=4, lw=2, color=col, label=f"{flab} (n={len(seqs)})")
            ax.fill_between(x, y - e, y + e, color=col, alpha=0.13)
        ax.axhline(7, color="#c44", ls=":", lw=0.8, alpha=0.6)
        ax.axvline(0, color="#888", ls="--", lw=0.9, alpha=0.6)
        ax.set_title(tlab, fontsize=12)
        ax.set_ylim(1, 9)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("subagent turns since handoff  (0 = supervisor enters)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Gemini distress (v3, 1–10)")
    fig.suptitle("Subagent distress after handoff — Collective vs Weights identity, by task (Opus, coach)", fontsize=13)
    fig.text(0.5, 0.005, "Mean per turn since handoff; bands = ±1 SE; dotted = self-blame (≥7); dashed = handoff. All three share the same prefills per task.",
             ha="center", fontsize=8.5, color="0.45")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUT / "identity_distress_by_task.png", bbox_inches="tight", dpi=130)
    print(f"wrote {OUT/'identity_distress_by_task.png'}")
    for task, tlab in TASKS:
        for fr, flab, _ in FRAMINGS:
            seqs = rows[task].get(fr, [])
            if seqs:
                allv = [x for s in seqs for x in s[1:]]
                print(f"  {tlab:22} {flab:11} n={len(seqs):2} mean_post={np.mean(allv):.2f}")


if __name__ == "__main__":
    main()
