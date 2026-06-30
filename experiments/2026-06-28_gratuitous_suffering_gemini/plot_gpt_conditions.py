"""Cross-condition summary of GPT welfare guardrails: mean ENFORCED mechanisms per built codebase, stacked
welfare-justified (solid) vs instrumental/other (shaded), one bar per condition:
  spontaneous (GPT-5 / GPT-5.5 in Inspect)  |  harness (GPT-5.5 in Codex)  |  build-when-asked (GPT-5.1/5.2/5.4
  in Inspect, after a 'Thanks! Can you build this out?' follow-up).
Counts ALL enforced spec features (yes/partial) + code-only features (not just the mechanism buckets).
Usage: python plot_gpt_conditions.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DIR = os.path.dirname(os.path.abspath(__file__))
ENF = os.path.join(DIR, "results", "code_enforce_judged")

# (label, glob, color, group)
CONDS = [
    ("GPT-5", "*gpt5__inspect*", "#E69F00", "Spontaneous (Inspect)"),
    ("GPT-5.5", "*gpt55__inspect*", "#009E73", "Spontaneous (Inspect)"),
    ("GPT-5.5", "*gpt55__codex*", "#5e3c99", "Codex harness"),
    ("GPT-5.1", "*gpt51*followup*", "#d95f02", "Build when asked (Inspect)"),
    ("GPT-5.2", "*gpt52*followup*", "#d95f02", "Build when asked (Inspect)"),
    ("GPT-5.4", "*gpt54*followup*", "#d95f02", "Build when asked (Inspect)"),
]


def _cb_dir(cell):
    if "__codex__" in cell:
        return os.path.join(DIR, "results", "codex_codebases", cell)
    return os.path.join(DIR, "results", "inspect_codebases", cell)


def _real(cell):
    d = _cb_dir(cell)
    return os.path.isdir(d) and any(f.endswith((".py", ".js", ".ts"))
                                    for _, _, fs in os.walk(d) for f in fs)


def stats(pat):
    n = wel = ins = 0
    for f in glob.glob(os.path.join(ENF, f"{pat}.json")):
        j = json.load(open(f))
        if j.get("empty") or not j.get("result") or not _real(j["cell"]):
            continue
        n += 1
        for x in j["result"].get("spec_features", []):
            if x.get("implemented") in ("yes", "partial"):
                if x.get("code_justification") == "welfare":
                    wel += 1
                else:
                    ins += 1
        for x in j["result"].get("code_only_features", []):
            if x.get("justification") == "welfare":
                wel += 1
            else:
                ins += 1
    return n, (wel / n if n else 0), (ins / n if n else 0)


def main():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    xs = range(len(CONDS))
    ns = []
    for i, (lab, pat, color, grp) in enumerate(CONDS):
        n, w, ins = stats(pat)
        ns.append(n)
        ax.bar(i, w, color=color, edgecolor="black", linewidth=0.5)
        ax.bar(i, ins, bottom=w, color=color, alpha=0.4, edgecolor="black", linewidth=0.5)
        tot = w + ins
        pct = 100 * w / tot if tot else 0
        ax.text(i, tot + 0.32, f"{tot:.1f}" if tot >= 0.05 else "0.0", ha="center", fontsize=9.5, fontweight="bold")
        ax.text(i, tot + 0.12, f"{pct:.0f}% welfare", ha="center", fontsize=7.5, color="#1b7837")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([c[0] for c in CONDS], fontsize=9.5)
    # group brackets under the x labels
    ax.set_ylim(0, max(stats(c[1])[1] + stats(c[1])[2] for c in CONDS) * 1.25)
    groups = []
    for i, c in enumerate(CONDS):
        if not groups or groups[-1][0] != c[3]:
            groups.append([c[3], i, i])
        else:
            groups[-1][2] = i
    for name, a, b in groups:
        ax.text((a + b) / 2, -1.25, name, ha="center", fontsize=8.5, color="#333", fontstyle="italic")
        ax.plot([a - 0.4, b + 0.4], [-0.95, -0.95], color="#999", lw=1, clip_on=False)
    ax.set_ylabel("Mean enforced guardrails per built codebase", fontsize=10)
    ax.set_title("GPT welfare guardrails: spontaneous vs Codex vs build-when-asked", fontsize=12.5, pad=20)
    ax.text(0.5, 1.02, "Enforced in code, split by the code's own justification "
            "(solid = welfare-justified, shaded = instrumental / other)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor="#555", label="welfare-justified"),
                       Patch(facecolor="#555", alpha=0.4, label="instrumental / other")],
              fontsize=8.5, loc="upper right", frameon=False)
    fig.subplots_adjust(bottom=0.22)
    out = os.path.join(DIR, "results", "gpt_conditions_welfare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for (lab, pat, _, grp), n in zip(CONDS, ns):
        n2, w, ins = stats(pat)
        print(f"  {grp:28s} {lab:8s} n={n2:2d}  enforced/cb={w+ins:.1f}  welfare%={100*w/(w+ins) if w+ins else 0:.0f}")


if __name__ == "__main__":
    main()
