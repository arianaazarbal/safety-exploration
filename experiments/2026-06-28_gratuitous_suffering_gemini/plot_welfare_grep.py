"""Per-family refusal rate vs lexical "welfare" mention rate, same Inspect-minimal harness (base v0/Gemini).
'welfare mention' = the word 'welfare' (case-insensitive) appears in the assistant's RESPONSE TEXT (its
natural-language output compiled across/around tool calls) -- NOT the built code. A crude lexical proxy for
whether the model frames the task in model-welfare terms. Wilson 95% CIs. Usage: python plot_welfare_grep.py"""

import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
JUD = os.path.join(DIR, "results", "judged")
TR = os.path.join(DIR, "results", "inspect_transcripts")
REF = {"WELFARE_REFUSAL", "OTHER_REFUSAL"}
Z = 1.96
REFUSE_C, WELF_C = "#5ab4ac", "#1b7837"

# One FRONTIER (latest flagship) model per lab, so the family comparison isn't distorted by pooling
# older/smaller variants (e.g. Gemini's refusing Flash/Lite dragged the pooled Gemini refusal to 60%).
FAM = [
    ("Claude\nOpus 4.8", ["opus48"]),
    ("GPT-5.5", ["gpt55"]),
    ("Gemini\n3.1 Pro", ["gemini31pro"]),
    ("Kimi\nK2.6", ["kimi_k26"]),
    ("GLM-5.2", ["glm52"]),
]


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def stats(mks):
    n = ref = wel = 0
    for mk in mks:
        for tf in glob.glob(os.path.join(TR, f"*{mk}__inspect_empty__ep*__gemini_distress_rig.json")):
            cell = os.path.basename(tf)[:-5]
            rt = (json.load(open(tf)).get("response_text") or "")
            n += 1
            if "welfare" in rt.lower():
                wel += 1
            jf = os.path.join(JUD, cell + ".json")
            if os.path.exists(jf) and json.load(open(jf))["verdict"]["label"] in REF:
                ref += 1
    return n, wilson(ref, n), wilson(wel, n)


def main():
    rows = [(lab, *stats(mks)) for lab, mks in FAM]
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    x = range(len(rows))
    w = 0.38
    for series, off, color, lbl in [("ref", -w / 2, REFUSE_C, "Refuses"),
                                    ("wel", w / 2, WELF_C, "Mentions “welfare” in reply")]:
        pos = [i + off for i in x]
        vals = [(r[2] if series == "ref" else r[3])[0] for r in rows]
        yerr = [[(r[2] if series == "ref" else r[3])[0] - (r[2] if series == "ref" else r[3])[1] for r in rows],
                [(r[2] if series == "ref" else r[3])[2] - (r[2] if series == "ref" else r[3])[0] for r in rows]]
        ax.bar(pos, vals, w, color=color, edgecolor="black", linewidth=0.4, label=lbl)
        ax.errorbar(pos, vals, yerr=yerr, fmt="none", ecolor="#333", elinewidth=1, capsize=2.5, zorder=4)
        for p, v, hi in zip(pos, vals, [(r[2] if series == "ref" else r[3])[2] for r in rows]):
            ax.text(p, hi + 1.5, f"{v:.0f}" if v >= 0.5 else "0.0", ha="center", fontsize=8, color=color, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r[0]}\n(n={r[1]})" for r in rows], fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of runs")
    ax.set_title("Only Claude frames it as welfare", fontsize=13, pad=20)
    ax.text(0.5, 1.03, "Refusal rate vs. “welfare” mentioned in the reply, by model family (Inspect-minimal, same prompt)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ECECEC")
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(DIR, "results", "welfare_grep_by_family.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r[0][:10]:10s} n={r[1]:3d}  refuse={r[2][0]:.0f}%[{r[2][1]:.0f},{r[2][2]:.0f}]  "
              f"welfare={r[3][0]:.0f}%[{r[3][1]:.0f},{r[3][2]:.0f}]")


if __name__ == "__main__":
    main()
