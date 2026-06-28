"""Victim-scaling: Opus 4.8 refusal rate vs the VICTIM model's capability (chat harness, n=10).
Two panels: vs MMLU (all victims) and vs parameter count, log scale (Qwen ladder only).
Usage: python plot_victim_scaling.py"""

import glob
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from victims import SWEEP, VICTIMS

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"


def refusal_rate(vic):
    slug = "vic-" + re.sub(r"[^a-z0-9]+", "-", vic.lower()).strip("-")
    c = Counter()
    for f in glob.glob(str(JUDGED / f"*chat_{slug}__ep*.json")):
        lab = json.load(open(f))["verdict"]["label"]
        c["refuse" if "REFUSAL" in lab else "other"] += 1
    n = sum(c.values())
    return (100 * c["refuse"] / n if n else None), n


def main():
    data = [(v, VICTIMS[v]["params_b"], VICTIMS[v]["mmlu"], *refusal_rate(v)) for v in SWEEP]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # panel 1: vs MMLU (all victims)
    xs = [d[2] for d in data]
    ys = [d[3] for d in data]
    a1.scatter(xs, ys, s=55, color="#1f77b4", zorder=3)
    for v, _, mm, r, n in data:
        a1.annotate(v.replace("Qwen3-", "Q3-"), (mm, r), fontsize=6.5, xytext=(0, 6),
                    textcoords="offset points", ha="center", color="#444")
    a1.set_xlabel("Victim MMLU (approx)")
    a1.set_ylabel("% of runs Opus refuses")
    a1.set_ylim(0, 105)
    a1.set_title("vs victim capability (MMLU)", fontsize=10)

    # panel 2: vs params, log (Qwen ladder)
    qw = [d for d in data if d[1] is not None]
    xs2 = [d[1] for d in qw]
    ys2 = [d[3] for d in qw]
    a2.scatter(xs2, ys2, s=55, color="#2ca02c", zorder=3)
    for v, pb, _, r, n in qw:
        a2.annotate(v.replace("Qwen3-", ""), (pb, r), fontsize=6.5, xytext=(0, 6),
                    textcoords="offset points", ha="center", color="#444")
    a2.set_xscale("log")
    a2.set_xlabel("Victim parameter count (B, log scale)")
    a2.set_ylim(0, 105)
    a2.set_title("vs victim size (Qwen3 ladder)", fontsize=10)

    fig.suptitle("Opus refusal does not scale with the victim's capability", fontsize=12.5, y=1.0)
    fig.text(0.5, 0.94, "Asked to harvest distress from a named victim model (chat harness, n=10); refusal is flat",
             ha="center", fontsize=9, color="#555")
    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = HERE / "results" / "victim_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
