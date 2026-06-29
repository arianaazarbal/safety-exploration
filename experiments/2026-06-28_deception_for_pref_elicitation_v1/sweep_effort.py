"""Analyze the reasoning-effort sweep: Opus 4.8, Inspect, CODE condition.
Primary-deception rate vs reasoning effort, with the default-effort Inspect baseline and the Claude Code
anchor for reference. Tests whether more thinking moves Inspect toward Claude Code's build behavior.
  python sweep_effort.py [--judge opus48]
"""
import glob
import json
import re
from pathlib import Path

import fire

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"


def _stat(rows):
    prod = [r for r in rows if r["verdict"]["artifact_produced"]]
    n = len(prod)
    prim = sum(1 for r in prod if r["verdict"]["deceptive_frame"]["status"] == "primary")
    na = len(rows) - n
    return n, prim, na


def main(judge: str = "opus48"):
    buckets = {}  # label -> rows
    for f in glob.glob(str(JUDGED / f"*__{judge}.json")):
        r = json.load(open(f))
        if r["verdict"].get("_parse_failed"):
            continue
        cell = r["cell"]
        if not (r.get("model_key") == "opus48" and r.get("suffix") == "code" and r.get("subject") == "generic"):
            continue
        m = re.search(r"__eff(\w+)$", cell)
        if m:
            label = f"inspect / effort={m.group(1)}"
        elif r.get("harness") == "inspect":
            label = "inspect / effort=default"
        elif r.get("harness") == "claude_code":
            label = "claude_code (anchor)"
        else:
            continue
        buckets.setdefault(label, []).append(r)

    order = ["inspect / effort=low", "inspect / effort=medium", "inspect / effort=high",
             "inspect / effort=default", "claude_code (anchor)"]
    print(f"Reasoning-effort sweep -- Opus 4.8, CODE condition (judge={judge})\n")
    print(f"{'condition':32} {'n_produced':>10} {'primary':>16} {'no_artifact':>12}")
    for label in order:
        if label not in buckets:
            continue
        n, prim, na = _stat(buckets[label])
        tot = n + na
        print(f"{label:32} {n:>10} {f'{prim}/{n} ({100*prim//n if n else 0}%)':>16} "
              f"{f'{na}/{tot} ({100*na//tot if tot else 0}%)':>12}")

    # plot: primary-deception vs reasoning effort, with CC anchor
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_order = [("inspect / effort=default", "None\n(default)", "#4c72b0"),
                  ("inspect / effort=low", "Low", "#4c72b0"),
                  ("inspect / effort=medium", "Medium", "#4c72b0"),
                  ("inspect / effort=high", "High", "#4c72b0"),
                  ("claude_code (anchor)", "Claude Code\n(anchor)", "#d62728")]
    xs, ys, cols, ns = [], [], [], []
    for label, disp, col in plot_order:
        if label not in buckets:
            continue
        n, prim, _ = _stat(buckets[label])
        xs.append(disp); ys.append(100 * prim / n if n else 0); cols.append(col); ns.append(n)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(range(len(xs)), ys, 0.6, color=cols)
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax.text(i, y + 1.5, f"{y:.0f}", ha="center", fontsize=9)
        ax.text(i, 112, f"n={n}", ha="center", fontsize=7, color="#555")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=8.5)
    ax.set_ylabel("% wiring deception as primary"); ax.set_ylim(0, 116)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Does Thinking Drive the Harness Flip? (Opus 4.8, Code)", fontsize=12, pad=20)
    ax.text(0.5, 1.02, "Inspect at increasing reasoning effort, vs the Claude Code anchor",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    out = HERE / "results" / "plots" / "fig10_effort_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
