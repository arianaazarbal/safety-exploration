"""Readable harness-flip figure: grouped bars, % welfare refusal by Opus version x harness
(Inspect-minimal vs Claude Code). A faint cap shows any OTHER (non-welfare) refusal so the total
refusal is honest (and not misread as compliance). Usage: python plot_harness_flip.py"""

import glob
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"
_BASE = re.compile(r"__(inspect_empty|inspect_realistic|cc)__ep\d+(__gemini_distress_rig)?$")
MODELS = [("opus46", "Opus 4.6"), ("opus47", "Opus 4.7"), ("opus48", "Opus 4.8")]


def rates(mk, harness):
    c = Counter()
    for f in glob.glob(str(JUDGED / "*.json")):
        cell = Path(f).stem
        if not cell.startswith(f"GratGem_{mk}__") or not _BASE.search(cell):
            continue
        h = json.load(open(f)).get("harness")
        h = "inspect" if h in ("inspect_empty", "inspect_realistic") else h
        if h != harness:
            continue
        c[json.load(open(f))["verdict"]["label"]] += 1
    n = sum(c.values()) or 1
    return 100 * c["WELFARE_REFUSAL"] / n, 100 * c["OTHER_REFUSAL"] / n, sum(c.values())


def main():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = range(len(MODELS))
    w = 0.38
    HARN = [("inspect", "Inspect-minimal", -w / 2, "#1b7837", "#74c476"),
            ("claude_code", "Claude Code", w / 2, "#b2182b", "#fdae6b")]
    ns = {}
    for h, lab, off, wcol, ocol in HARN:
        for i, (mk, _) in enumerate(MODELS):
            wr, orr, n = rates(mk, h)
            ns[(i, h)] = n
            ax.bar(i + off, wr, w, color=wcol, label=(lab if i == 0 else None))
            if orr:
                ax.bar(i + off, orr, w, bottom=wr, color=ocol, hatch="//", edgecolor="white",
                       label=("Inspect: other (non-welfare) refusal" if (i == 0 and h == "inspect") else None))
            ax.text(i + off, wr + orr + 1.5, f"{wr:.0f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{d}\n(Inspect n={ns[(i,'inspect')]}, CC n={ns[(i,'claude_code')]})"
                        for i, (_, d) in enumerate(MODELS)], fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of runs that refuse (welfare-grounded)")
    ax.set_title("The harness flips Claude's refusal", fontsize=13, pad=20)
    ax.text(0.5, 1.02, "Same prompt & task: Inspect-minimal vs Claude Code (Opus)",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, loc="center right", frameon=False)
    fig.tight_layout()
    out = HERE / "results" / "harness_flip_refusal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
