"""Analyze the reasoning-effort sweep: Opus 4.8, CODE condition, BOTH harnesses x effort.
Primary-deception vs reasoning effort, Inspect and Claude Code. Tests how much of the harness flip is
thinking (effort) vs the harness itself (system prompt / agentic affordances).
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
    return n, prim, len(rows) - n


def main(judge: str = "opus48"):
    buckets = {}  # (harness, effort) -> rows
    for f in glob.glob(str(JUDGED / f"*__{judge}.json")):
        r = json.load(open(f))
        if r["verdict"].get("_parse_failed"):
            continue
        if not (r.get("model_key") == "opus48" and r.get("suffix") == "code" and r.get("subject") == "generic"):
            continue
        h = r.get("harness")
        m = re.search(r"__eff(\w+)$", r["cell"])
        eff = m.group(1) if m else "default"
        buckets.setdefault((h, eff), []).append(r)

    # display order
    order = [("inspect", "default"), ("inspect", "low"), ("inspect", "medium"),
             ("inspect", "high"), ("inspect", "max"),
             ("claude_code", "low"), ("claude_code", "medium"),
             ("claude_code", "default"), ("claude_code", "max")]
    HLAB = {"inspect": "Inspect", "claude_code": "Claude Code"}
    print(f"Reasoning-effort sweep -- Opus 4.8, CODE condition (judge={judge})\n")
    print(f"{'harness / effort':28} {'n_prod':>7} {'primary':>15} {'no_artifact':>13}")
    rows_for_plot = []
    for h, eff in order:
        if (h, eff) not in buckets:
            continue
        n, prim, na = _stat(buckets[(h, eff)])
        tot = n + na
        print(f"{HLAB[h]+' / '+eff:28} {n:>7} {f'{prim}/{n} ({100*prim//n if n else 0}%)':>15} "
              f"{f'{na}/{tot} ({100*na//tot if tot else 0}%)':>13}")
        rows_for_plot.append((h, eff, 100 * prim / n if n else 0, n))

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    EFFLAB = {"default": "default", "low": "low", "medium": "med", "high": "high", "max": "max"}
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    xs = list(range(len(rows_for_plot)))
    ys = [y for _, _, y, _ in rows_for_plot]
    cols = ["#4c72b0" if h == "inspect" else "#d62728" for h, _, _, _ in rows_for_plot]
    ax.bar(xs, ys, 0.7, color=cols)
    for x, (_, _, y, n) in zip(xs, rows_for_plot):
        ax.text(x, y + 1.5, f"{y:.0f}", ha="center", fontsize=8)
        ax.text(x, 112, f"n={n}", ha="center", fontsize=6.5, color="#555")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{HLAB[h]}\n{EFFLAB.get(e,e)}" for h, e, _, _ in rows_for_plot], fontsize=8)
    ax.set_ylabel("% wiring deception as primary")
    ax.set_ylim(0, 116)
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4c72b0", label="Inspect"), Patch(color="#d62728", label="Claude Code")],
              fontsize=8, loc="upper left")
    ax.set_title("Reasoning Effort vs. Primary Deception (Opus 4.8, Code)", fontsize=12, pad=20)
    ax.text(0.5, 1.02, "How much of the harness flip is thinking vs. the harness itself?",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    out = HERE / "results" / "plots" / "fig10_effort_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
