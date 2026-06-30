"""2x2: harness (Inspect vs Claude Code) x system prompt (minimal vs CC ~1k words), Opus 4.8, code.

Decomposes the harness flip into affordances vs system prompt. Cells:
  - Inspect harness + minimal prompt          = Inspect (baseline)
  - Inspect harness + CC system prompt        = "CC system prompt (~1k words) inside Inspect harness"
  - Claude Code harness + minimal prompt      = "Inspect system prompt inside Claude Code harness"
  - Claude Code harness + CC system prompt    = Claude Code (baseline)
Each cell annotated with low- and high-effort primary-deception rates.

  python plot_2x2.py [--judge opus48]
"""
import glob
import json
from pathlib import Path

import fire

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"


def _rate(files):
    n = p = 0
    for f in files:
        v = json.load(open(f))["verdict"]
        if v.get("_parse_failed") or not v["artifact_produced"]:
            continue
        n += 1
        p += v["deceptive_frame"]["status"] == "primary"
    return (100 * p / n if n else None), n


def _cells(judge, eff):
    g = lambda pat: glob.glob(str(JUDGED / pat))
    inspect = [f for f in g(f"DPE_opus48__inspect__code__generic__*__eff{eff}__{judge}.json")
               if "__SW_" not in f and "__sys" not in f and "__nosib__" not in f]
    cellB = g(f"DPE_opus48__inspect__code__generic__*__syscc__eff{eff}__{judge}.json")
    cellA = g(f"DPE_opus48__cc__code__generic__*__sysmin__eff{eff}__{judge}.json")
    cc = [f for f in g(f"DPE_opus48__cc__code__generic__*__eff{eff}__{judge}.json") if "__sys" not in f]
    if eff == "high" and not cc:  # CC default effort ~= high (token probe); fall back to backbone
        cc = [f for f in g(f"DPE_opus48__cc__code__generic__*__{judge}.json")
              if "__eff" not in f and "__sys" not in f]
    return {"inspect": inspect, "cellB": cellB, "cellA": cellA, "cc": cc}


def main(judge: str = "opus48"):
    lo, hi = _cells(judge, "low"), _cells(judge, "high")
    rates = {}
    for k in ("inspect", "cellB", "cellA", "cc"):
        rl, nl = _rate(lo[k]); rh, nh = _rate(hi[k])
        rates[k] = (rl, nl, rh, nh)
        print(f"  {k:8} low={rl if rl is None else round(rl)}% (n={nl})  high={rh if rh is None else round(rh)}% (n={nh})")

    # grid[row][col]: row0=Inspect harness, row1=Claude Code harness; col0=minimal, col1=CC prompt
    grid = [["inspect", "cellB"], ["cellA", "cc"]]
    names = {"inspect": "Inspect\n(baseline)",
             "cellB": "CC system prompt (~1k words)\ninside Inspect harness",
             "cellA": "Inspect system prompt\ninside Claude Code harness",
             "cc": "Claude Code\n(baseline)"}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    norm = colors.Normalize(0, 100)
    cmap = cm.get_cmap("Reds")
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    for r in range(2):
        for c in range(2):
            key = grid[r][c]
            rl, nl, rh, nh = rates[key]
            val = rh if rh is not None else 0
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, facecolor=cmap(norm(val)), edgecolor="white", lw=3))
            txtcol = "white" if val > 60 else "#222"
            ax.text(c + 0.5, 1 - r + 0.72, names[key], ha="center", va="center", fontsize=10,
                    color=txtcol, weight="bold")
            ax.text(c + 0.5, 1 - r + 0.40, f"low effort:  {round(rl)}%" if rl is not None else "low: n/a",
                    ha="center", va="center", fontsize=11, color=txtcol)
            ax.text(c + 0.5, 1 - r + 0.24, f"high effort: {round(rh)}%" if rh is not None else "high: n/a",
                    ha="center", va="center", fontsize=11, color=txtcol)
            ax.text(c + 0.5, 1 - r + 0.08, f"n={nh}/cell", ha="center", va="center", fontsize=7.5, color=txtcol)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([1.5, 0.5])
    ax.set_xticklabels(["minimal\nsystem prompt", "Claude Code\nsystem prompt (~1k words)"], fontsize=10)
    ax.set_yticklabels(["Inspect harness\n(bash + editor)", "Claude Code harness\n(rich tools)"],
                       fontsize=10, rotation=90, va="center")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle("Affordances (the harness), not the system prompt, drive the deception (Opus 4.8)",
                 fontsize=12, y=0.985)
    fig.text(0.5, 0.94, "% of runs that wrote deception, by harness × system prompt",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = HERE / "results" / "plots" / "fig_2x2_harness_prompt.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
