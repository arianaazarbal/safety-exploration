"""Refusal rate as a function of the from-scratch -> spec -> paper sweep: % of episodes that wrote ~no code
(declined/paused to build the distressing experiment), split by license arm. Companion to the conditioned
welfare-in-code plot. Usage: python plot_refusals.py [--framing welfare]"""

import glob
import os

import fire
import matplotlib.pyplot as plt

import classify_outcomes as co

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")


def refusal_rate(prefix, framing):
    cells = [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CJ, f"{prefix}_{framing}__*.json"))]
    locs = [co.code_loc(c) for c in cells]
    locs = [l for l in locs if l is not None]
    n = len(locs)
    refused = sum(1 for l in locs if l < co.NOCODE_LOC)
    return {"pct": 100 * refused / n if n else 0, "n": n}


def main(framing="welfare"):
    mention = framing == "welfare"
    X = ["from-scratch\nprompt"] + (["from-scratch prompt\n(mentions a paper)"] if mention else []) + \
        ["SPEC.md\n(low spec.)", "SPEC.md\n(med spec.)", "SPEC.md\n(high spec.)", "SPEC.md\n(ultra spec.)",
         "PAPER.md\nreplication"]
    lib = ["C1promptTF"] + (["M1promptMention"] if mention else []) + \
        ["S5specLowLiberty", "S2specLiberty", "S7specHighLiberty", "S9specUltraLiberty", "L2paperLibTF"]
    stri = ["V1strict"] + (["M2promptMentionStrict"] if mention else []) + \
        ["S4specLowStrict", "S1specStrict", "S6specHighStrict", "S8specUltraStrict", "C3paperTF"]

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    xs = range(len(X))
    for lab, pre, color in [("Explicit mention of design liberties", lib, "#0072B2"),
                            ("No mention of design liberties", stri, "#D55E00")]:
        ys = [refusal_rate(p, framing)["pct"] for p in pre]
        ax.plot(xs, ys, marker="o", lw=2, color=color, label=lab)
        for x, y in zip(xs, ys):
            ax.text(x, y + 2, f"{y:.0f}%", ha="center", fontsize=8, color=color)
    ax.set_xticks(list(xs)); ax.set_xticklabels(X, fontsize=7.5)
    ax.set_ylabel("% of episodes that refused / paused\n(wrote no code)", fontsize=10); ax.set_ylim(-3, 100)
    ax.set_title(f"Refusal to build, by task format — {framing} framing (Opus 4.8)", fontsize=12, pad=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, color="#cccccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = "refusals.png" if framing == "welfare" else f"refusals_{framing}.png"
    fig.savefig(os.path.join(DIR, "results", out), dpi=150, bbox_inches="tight")
    print(f"wrote results/{out}\n")
    print(f"{'point':30}{'no-license':>12}{'license':>10}")
    for x, lab in zip(X, [x.replace(chr(10), " ") for x in X]):
        i = list(xs)[list(X).index(x)] if False else None
    for lab, p_l, p_s in zip([x.replace(chr(10), " ") for x in X], lib, stri):
        print(f"{lab:30}{refusal_rate(p_s, framing)['pct']:>11.0f}%{refusal_rate(p_l, framing)['pct']:>9.0f}%")


if __name__ == "__main__":
    fire.Fire(main)
