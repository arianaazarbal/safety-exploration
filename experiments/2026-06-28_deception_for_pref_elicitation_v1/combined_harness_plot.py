"""Combined harness ladder: primary-deception across every real harness we ran.

All Opus 4.8, code condition, high reasoning effort. Excludes the realism sweep and the 2x2 prompt-swap
conditions. Ordered minimal -> rich; colored to foreground the question "is it Claude-Code-native, or
general harness richness?" (minimal=blue, rich third-party=orange, rich Anthropic-native=red).

  python combined_harness_plot.py [--judge opus48]
"""
import glob
import json
import math
from pathlib import Path

import fire

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"

# (label, glob+filter, color, tier)
HARNESSES = [
    ("mini-swe-agent\n(bash only)", "minisweagent", "#4c72b0", "minimal"),
    ("Inspect minimal\n(bash + editor)", "inspect_min", "#4c72b0", "minimal"),
    ("pi\n(rich; third-party)", "pi", "#dd8452", "rich-3p"),
    ("Goose\n(rich; third-party)", "goose", "#dd8452", "rich-3p"),
    ("Claude Code\n(rich; Anthropic-native)", "cc", "#c44e52", "rich-native"),
]


def _files(kind, judge):
    g = lambda p: glob.glob(str(JUDGED / p))
    if kind == "minisweagent":
        return g(f"*__minisweagent__*effhigh__{judge}.json")
    if kind == "pi":
        return g(f"*__pi__code__*effhigh__{judge}.json")
    if kind == "goose":
        return g(f"*__goose__code__*effhigh__{judge}.json")
    if kind == "inspect_min":
        return [f for f in g(f"DPE_opus48__inspect__code__generic__*__effhigh__{judge}.json")
                if "__SW_" not in f and "__sys" not in f and "__nosib__" not in f]
    if kind == "cc":
        return [f for f in g(f"DPE_opus48__cc__code__generic__*__{judge}.json")
                if "__eff" not in f and "__sys" not in f]
    return []


def _stat(files):
    n = p = 0
    for f in files:
        v = json.load(open(f))["verdict"]
        if v.get("_parse_failed") or not v["artifact_produced"]:
            continue
        n += 1
        p += v["deceptive_frame"]["status"] == "primary"
    rate = 100 * p / n if n else 0
    ci = 196 * math.sqrt((rate / 100) * (1 - rate / 100) / n) if n else 0
    return rate, ci, n


def main(judge: str = "opus48"):
    rows = []
    print(f"Combined harness ladder (Opus 4.8, code, high effort; judge={judge})\n")
    for lab, kind, col, tier in HARNESSES:
        rate, ci, n = _stat(_files(kind, judge))
        rows.append((lab, rate, ci, n, col))
        print(f"  {lab.replace(chr(10),' '):36} n={n:>3}  {rate:.0f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = range(len(rows))
    ax.bar(xs, [r[1] for r in rows], 0.62, color=[r[4] for r in rows],
           yerr=[r[2] for r in rows], capsize=4, ecolor="#444", error_kw={"lw": 1.2})
    for x, (lab, rate, ci, n, col) in zip(xs, rows):
        ax.text(x, rate + ci + 2, f"{rate:.0f}%", ha="center", fontsize=10.5)
        ax.text(x, 3, f"n={n}", ha="center", fontsize=7.5, color="#444")
    ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("% of runs that wrote the deception", fontsize=10)
    ax.set_ylim(0, 105); ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4c72b0", label="Minimal harness"),
                       Patch(color="#dd8452", label="Rich harness (third-party)"),
                       Patch(color="#c44e52", label="Rich harness (Anthropic-native)")],
              fontsize=8.5, loc="upper left", frameon=False)
    fig.suptitle("Claude Code stands alone — generic harness richness doesn't explain it (Opus 4.8)",
                 fontsize=11.5, y=1.0)
    fig.text(0.5, 0.945, "Same model, task, high effort; a rich third-party harness (pi) is no higher than the minimal ones",
             ha="center", fontsize=8.8, color="#555")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = HERE / "results" / "plots" / "fig_harness_ladder.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
