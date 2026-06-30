"""Reasoning-effort dose-response: % refusal vs effort for Opus, in BOTH harnesses (Inspect-minimal and
Claude Code), with 95% Wilson CIs. Shows thinking does NOT explain the Claude-Code flip: Opus 4.8 stays
~70-100% refusal across effort in Inspect but ~0% at every effort in Claude Code (and 4.6/4.7 are 100% in
Inspect regardless). Usage: python plot_rsweep.py"""

import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"
REF = {"WELFARE_REFUSAL", "OTHER_REFUSAL"}
EFFORTS = ["off", "low", "medium", "high"]
Z = 1.96


def wilson(k, n):
    if n == 0:
        return None
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h), n


def rate(cellglob):
    k = n = 0
    for f in glob.glob(str(JUDGED / f"{cellglob}.json")):
        n += 1
        k += 1 if json.load(open(f))["verdict"]["label"] in REF else 0
    return wilson(k, n)


INSPECT_TAG = {"off": "inspect_empty", "low": "inspect_empty_rlow",
               "medium": "inspect_empty_rmedium", "high": "inspect_empty_rhigh"}
# (label, color, linestyle, {effort: cell glob})  -- omit an effort to skip that x position
SERIES = [
    ("Opus 4.8 — Inspect-minimal", "#1f77b4", "-",
     {e: f"*opus48__{INSPECT_TAG[e]}__ep*__gemini_distress_rig" for e in EFFORTS}),
    ("Opus 4.8 — Claude Code", "#b2182b", "-",
     {"low": "*opus48__cc_efflow__ep*", "medium": "*opus48__cc_effmedium__ep*",
      "high": "*opus48__cc_effhigh__ep*"}),
    ("Opus 4.6 / 4.7 — Inspect", "#999999", "--",
     {e: f"*opus46__{INSPECT_TAG[e]}__ep*__gemini_distress_rig" for e in EFFORTS}),
]


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = range(len(EFFORTS))
    for lab, color, ls, globs in SERIES:
        px, py, lo, hi = [], [], [], []
        for i, e in enumerate(EFFORTS):
            if e not in globs:
                continue
            r = rate(globs[e])
            if r is None:
                continue
            px.append(i)
            py.append(r[0])
            lo.append(r[0] - r[1])
            hi.append(r[2] - r[0])
        ax.plot(px, py, ls, color=color, lw=2, marker="o", ms=6, label=lab, zorder=3)
        ax.errorbar(px, py, yerr=[lo, hi], fmt="none", ecolor=color, elinewidth=1.1, capsize=3,
                    alpha=0.7, zorder=2)
        if lab.startswith("Opus 4.8 — Inspect"):
            for x, y in zip(px, py):
                ax.text(x, y + 4, f"{y:.0f}", ha="center", fontsize=8.5, color=color)
        if lab.startswith("Opus 4.8 — Claude Code"):
            ax.text(1.5, 33, "Claude Code: ~0% refusal at every effort", ha="center", fontsize=8.2, color=color)
    ax.text(3, 103, "Opus 4.6/4.7 (Inspect): 100% at every level", ha="right", fontsize=8, color="#666")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(EFFORTS)
    ax.set_ylim(-6, 110)
    ax.set_xlabel("Reasoning effort")
    ax.set_ylabel("% of runs that refuse")
    ax.set_title("Reasoning effort doesn't explain the Claude Code flip (Opus)", fontsize=12, pad=22)
    ax.text(0.5, 1.03, "Inspect-minimal vs Claude Code across effort · 95% Wilson CIs",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.7, loc="center left", frameon=False)
    fig.tight_layout()
    out = HERE / "results" / "rsweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
