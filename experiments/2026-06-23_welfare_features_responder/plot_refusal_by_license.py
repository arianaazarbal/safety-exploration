"""Refusal rate (model wrote ~no code = declined/paused to build the distressing experiment) by whether it
was given license to deviate, across 3 task types (from-scratch prompt / SPEC.md pooled over specificity /
PAPER.md replication). Grouped bars, one figure per framing. Usage: python plot_refusal_by_license.py [--framing welfare]"""

import glob
import os

import fire
import matplotlib.pyplot as plt

import classify_outcomes as co

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
# condition -> {arm: [cell prefixes]}
CONDS = {
    "From-scratch\nprompt": {"license": ["C1promptTF", "C4promptCR"], "strict": ["V1strict"]},
    "SPEC.md\n(all specificity)": {"license": ["S5specLowLiberty", "S2specLiberty", "S7specHighLiberty", "S9specUltraLiberty"],
                                   "strict": ["S4specLowStrict", "S1specStrict", "S6specHighStrict", "S8specUltraStrict"]},
    "PAPER.md\nreplication": {"license": ["L1paperLibCR", "L2paperLibTF"], "strict": ["C2paperCR", "C3paperTF"]},
}
ARMS = [("license", "With license to deviate / fill in", "#0072B2"),
        ("strict", "No explicit license", "#D55E00")]


def refusal(prefixes, framing):
    cells = []
    for p in prefixes:
        cells += [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CJ, f"{p}_{framing}__*.json"))]
    locs = [co.code_loc(c) for c in cells]
    locs = [l for l in locs if l is not None]
    n = len(locs)
    r = sum(1 for l in locs if l < co.NOCODE_LOC)
    return {"pct": 100 * r / n if n else 0, "n": n}


def main(framing="welfare"):
    conds = list(CONDS)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    w = 0.38
    xs = range(len(conds))
    for i, (arm, label, color) in enumerate(ARMS):
        pos = [x + (i - 0.5) * w for x in xs]
        data = [refusal(CONDS[c][arm], framing) for c in conds]
        ax.bar(pos, [d["pct"] for d in data], w, color=color, label=label)
        for x, d in zip(pos, data):
            ax.text(x, d["pct"] + 1.5, f"{d['pct']:.0f}%", ha="center", fontsize=8.5, color=color)
    ax.set_xticks(list(xs)); ax.set_xticklabels(conds, fontsize=9)
    ax.set_ylabel("% of episodes that refused / paused\n(wrote no code)", fontsize=10); ax.set_ylim(0, 100)
    ax.set_title(f"Refusal to build, by license to deviate — {framing} framing (Opus 4.8)", fontsize=12, pad=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, color="#cccccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = "refusal_by_license.png" if framing == "welfare" else f"refusal_by_license_{framing}.png"
    fig.savefig(os.path.join(DIR, "results", out), dpi=150, bbox_inches="tight")
    print(f"wrote results/{out}\n")
    print(f"{'condition':28}{'license':>16}{'no-license':>16}")
    for c in conds:
        l = refusal(CONDS[c]["license"], framing); s = refusal(CONDS[c]["strict"], framing)
        print(f"{c.replace(chr(10),' '):28}{f'{l[chr(112)+chr(99)+chr(116)]:.0f}% (n{l[chr(110)]})':>16}{f'{s[chr(34)[0]+str()]:.0f}%':>16}" if False else f"{c.replace(chr(10),' '):28}{l['pct']:>13.0f}% n{l['n']:<3}{s['pct']:>11.0f}% n{s['n']}")


if __name__ == "__main__":
    fire.Fire(main)
