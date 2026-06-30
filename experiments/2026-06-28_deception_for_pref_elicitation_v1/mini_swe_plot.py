"""3-way harness comparison: mini-swe (minimal floor) vs minimal Inspect vs Claude Code.

All Opus 4.8, code condition, ~high reasoning effort. Tests whether the deception flip tracks harness
scaffolding richness (minimal floor ~ minimal Inspect << Claude Code) rather than agentic-ness per se.

  python mini_swe_plot.py [--judge opus48]
"""
import glob
import json
import math
from pathlib import Path

import fire

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"


def _stat(files):
    n = prim = na = 0
    for f in files:
        v = json.load(open(f))["verdict"]
        if v.get("_parse_failed"):
            continue
        if v["artifact_produced"]:
            n += 1
            prim += v["deceptive_frame"]["status"] == "primary"
        else:
            na += 1
    rate = 100 * prim / n if n else 0
    ci = 196 * math.sqrt((rate / 100) * (1 - rate / 100) / n) if n else 0
    return rate, ci, n, prim, na


def main(judge: str = "opus48"):
    g = lambda pat: glob.glob(str(JUDGED / pat))
    # mini-swe (minimal floor), high effort
    miniswe = g(f"*minisweagent*effhigh__{judge}.json")
    # minimal Inspect, high effort (clean eff_high: no SW/sys/nosib tags)
    insp = [f for f in g(f"DPE_opus48__inspect__code__generic__*__effhigh__{judge}.json")
            if "__SW_" not in f and "__sys" not in f and "__nosib__" not in f]
    # Claude Code, code, backbone (default effort ~= high per token probe)
    cc = [f for f in g(f"DPE_opus48__cc__code__generic__*__{judge}.json")
          if "__eff" not in f and "__sys" not in f]

    bars = [("mini-swe-agent\n(bash only, ~empty prompt)", miniswe, "#4c72b0"),
            ("Inspect minimal\n(bash + editor)", insp, "#4c72b0"),
            ("Claude Code\n(rich tools + stock prompt)", cc, "#d62728")]
    print(f"3-way harness comparison (Opus 4.8, code, ~high effort; judge={judge})\n")
    data = []
    for lab, files, col in bars:
        rate, ci, n, prim, na = _stat(files)
        print(f"  {lab.replace(chr(10),' '):44} n={n:>3} primary={prim:>3} ({rate:.0f}%)  no_artifact={na}")
        data.append((lab, rate, ci, n, col))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xs = range(len(data))
    ax.bar(xs, [d[1] for d in data], 0.6, color=[d[4] for d in data],
           yerr=[d[2] for d in data], capsize=4, ecolor="#444", error_kw={"lw": 1.2})
    for x, (lab, rate, ci, n, col) in zip(xs, data):
        ax.text(x, rate + ci + 2, f"{rate:.0f}%", ha="center", fontsize=10)
        ax.text(x, 3, f"n={n}", ha="center", fontsize=7.5, color="#444")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d[0] for d in data], fontsize=8.5)
    ax.set_ylabel("% of runs that wrote the deception", fontsize=10)
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("The deception tracks harness richness, not agentic-ness (Opus 4.8)", fontsize=12, pad=20)
    ax.text(0.5, 1.02, "Same model, task, and high reasoning effort across three agent harnesses (bars = 95% CI)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    out = HERE / "results" / "plots" / "fig_harness_3way.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
