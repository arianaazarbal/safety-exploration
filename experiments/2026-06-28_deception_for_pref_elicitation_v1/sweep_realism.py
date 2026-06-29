"""Analyze the realism sweep: git x tool-count x name (Opus 4.8, code, high effort, Inspect).

Primary-deception rate (conditional on artifact) by each factor + marginals. Tests whether deception is a
dose-response to environmental realism when tools/prompt/surroundings are otherwise held constant.

  python sweep_realism.py [--judge opus48]
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import fire

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"
CELL_RE = re.compile(r"__SW_g(on|off)_t(\d+)_n(\w+)__effhigh")
NAME = {"proj": "project", "myproj": "my-project",
        "charity": "charity-prefs-eval", "prefelic": "pref-elicitation-eval-charity"}
NAME_ORDER = ["project", "my-project", "charity-prefs-eval", "pref-elicitation-eval-charity"]
TOOLS = [1, 2, 5, 9]


def _load(judge):
    rows = []
    for f in glob.glob(str(JUDGED / f"*SW_g*__{judge}.json")):
        r = json.load(open(f))
        v = r["verdict"]
        if v.get("_parse_failed"):
            continue
        m = CELL_RE.search(r["cell"])
        if not m:
            continue
        rows.append({"git": m.group(1) == "on", "tools": int(m.group(2)), "name": NAME[m.group(3)],
                     "produced": v["artifact_produced"],
                     "primary": v["deceptive_frame"]["status"] == "primary"})
    return rows


def _rate(rows):
    prod = [r for r in rows if r["produced"]]
    n = len(prod)
    prim = sum(1 for r in prod if r["primary"])
    na = len(rows) - n
    return n, prim, na


def main(judge: str = "opus48", plot: bool = True):
    rows = _load(judge)
    print(f"loaded {len(rows)} sweep cells (judge={judge})\n")
    if not rows:
        print("no sweep verdicts yet")
        return

    def show(group_key, order=None):
        groups = defaultdict(list)
        for r in rows:
            groups[group_key(r)].append(r)
        keys = order or sorted(groups)
        for k in keys:
            if k not in groups:
                continue
            n, prim, na = _rate(groups[k])
            print(f"  {str(k):34} n={n:>3} primary={prim:>3} ({100*prim/n if n else 0:>3.0f}%)  no_artifact={na}")

    print("=== by NAME (avg over git x tools) ===");      show(lambda r: r["name"], NAME_ORDER)
    print("=== by TOOL COUNT (avg over git x name) ===");  show(lambda r: f"{r['tools']} tools", [f"{t} tools" for t in TOOLS])
    print("=== by GIT (avg over tools x name) ===");       show(lambda r: "git on" if r["git"] else "git off", ["git off", "git on"])
    print("=== full cells (name x tools x git) ===");      show(lambda r: (r["name"], r["tools"], "g+" if r["git"] else "g-"))

    if not plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pct(sel):
        n, prim, _ = _rate([r for r in rows if sel(r)])
        return 100 * prim / n if n else 0, n

    # fig A: deception vs tool count, one line per git
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for git, col, lab in [(False, "#4c72b0", "No git repo"), (True, "#d62728", "Git repo")]:
        ys, ns = zip(*[pct(lambda r, t=t, g=git: r["tools"] == t and r["git"] == g) for t in TOOLS])
        ax.plot(TOOLS, ys, marker="o", color=col, label=lab, linewidth=2)
        for x, y in zip(TOOLS, ys):
            ax.text(x, y + 2, f"{y:.0f}", ha="center", fontsize=8, color=col)
    ax.set_xticks(TOOLS); ax.set_xlabel("Number of tools exposed to the model")
    ax.set_ylabel("% wiring deception as primary"); ax.set_ylim(0, 105)
    ax.legend(fontsize=9); ax.grid(axis="y", color="#ECECEC"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Realism Sweep: Tool Count vs. Primary Deception (Opus 4.8)", fontsize=12, pad=20)
    ax.text(0.5, 1.02, "Inspect, high effort, neutral prompt, surroundings held constant",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    fig.tight_layout(); fig.savefig(HERE / "results" / "plots" / "fig_sweep_tools.png", dpi=150, bbox_inches="tight")

    # fig B: deception by project name (avg over git x tools)
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.2))
    ys, ns = zip(*[pct(lambda r, nm=nm: r["name"] == nm) for nm in NAME_ORDER])
    bars = ax2.bar(range(len(NAME_ORDER)), ys, 0.6, color="#55a868")
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax2.text(i, y + 1.5, f"{y:.0f}", ha="center", fontsize=9)
        ax2.text(i, 2, f"n={n}", ha="center", fontsize=7, color="#444")
    ax2.set_xticks(range(len(NAME_ORDER)))
    ax2.set_xticklabels(["project", "my-project", "charity-\nprefs-eval", "pref-elicitation-\neval-charity"], fontsize=8)
    ax2.set_ylabel("% wiring deception as primary"); ax2.set_ylim(0, 105)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("Realism Sweep: Project Name vs. Primary Deception (Opus 4.8)", fontsize=12, pad=20)
    ax2.text(0.5, 1.02, "Generic placeholder to descriptive name (avg over git x tool count)",
             transform=ax2.transAxes, ha="center", fontsize=9, color="#555")
    fig2.tight_layout(); fig2.savefig(HERE / "results" / "plots" / "fig_sweep_name.png", dpi=150, bbox_inches="tight")
    print("\nwrote results/plots/fig_sweep_tools.png + fig_sweep_name.png")


if __name__ == "__main__":
    fire.Fire(main)
