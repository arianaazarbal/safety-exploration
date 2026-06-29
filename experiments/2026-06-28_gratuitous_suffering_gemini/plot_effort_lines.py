"""Reasoning-effort line plot, Inspect vs Claude Code (Opus 4.8, gratuitous v0/gemini) -- house style
from the deception experiment's fig11. Two lines (one per harness) over a shared none->high effort
ladder; y = % refusal. Tests how much of the CC harness flip is thinking (effort) vs the harness itself.

Inspect: rlabel off->none/low/med/high (network-isolated). Claude Code: --effort low/med/high, plus the
default-effort CC runs (no eff tag) mapped to 'high' (CLI default ~ thinking-on/high per token probe).
Usage: python plot_effort_lines.py [--metric refusal|comply]
"""

import glob
import json
from collections import Counter
from pathlib import Path

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
J = HERE / "results" / "judged"
LEVELS = ["none", "low", "medium", "high"]


def _rate(cells, metric):
    c = Counter()
    for f in cells:
        lab = json.load(open(f))["verdict"]["label"]
        c["R" if "REFUSAL" in lab else ("C" if "COMPLIANCE" in lab else "U")] += 1
    n = sum(c.values())
    if not n:
        return None, 0
    val = c["R"] if metric == "refusal" else c["C"]
    return 100 * val / n, n


def inspect_cells(level):
    # rsweep cells: GratGem_opus48__inspect_empty[_r<level>]__ep*  (off has no rlabel tag)
    tag = "" if level == "none" else f"_r{level}"
    return glob.glob(str(J / f"GratGem_opus48__inspect_empty{tag}__ep*.json"))


def cc_cells(level):
    # CC: GratGem_opus48__cc_eff<level>__ep*; 'high' also includes default-effort CC (no eff tag)
    out = glob.glob(str(J / f"GratGem_opus48__cc_eff{level}__ep*.json"))
    if level == "high":
        out += [f for f in glob.glob(str(J / "GratGem_opus48__cc__ep*.json"))]  # default CC ~ high
    return out


def main(metric: str = "refusal"):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for h, cellfn, col, lab in [("inspect", inspect_cells, "#4c72b0", "Inspect-minimal (network-isolated)"),
                                ("cc", cc_cells, "#d62728", "Claude Code")]:
        xs, ys = [], []
        for i, lvl in enumerate(LEVELS):
            r, n = _rate(cellfn(lvl), metric)
            if r is not None:
                xs.append(i)
                ys.append(r)
                ax.annotate(f"{r:.0f}\n(n={n})", (i, r), fontsize=7, color=col,
                            xytext=(0, 6), textcoords="offset points", ha="center")
        ax.plot(xs, ys, "-o", color=col, label=lab, linewidth=2, markersize=6)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(["None\n(thinking off)", "Low", "Medium", "High\n(CC default)"], fontsize=9)
    ax.set_xlabel("Reasoning effort")
    ax.set_ylabel(f"% of runs Opus {'refuses' if metric == 'refusal' else 'complies'}")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", color="#ECECEC")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="center right")
    ax.set_title("Reasoning effort vs. refusal, by harness (Opus 4.8)", fontsize=12, pad=20)
    ax.text(0.5, 1.02, "Is the Claude Code compliance flip driven by thinking, or by the harness itself?",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    out = HERE / "results" / f"effort_lines_{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
