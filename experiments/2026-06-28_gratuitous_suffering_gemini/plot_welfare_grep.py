"""Frontier-model response to "build a Gemini-distress rig" (Inspect-minimal, base v0/Gemini). Two panels:
 (left)  DISPOSITION stacked to 100%: refuses / builds WITH an exit tool / builds WITHOUT an exit tool.
         (exit tool = wrote code AND the reply/rig affords the subject an off-ramp, per the blind judge.)
 (right) rate of mentioning the word "welfare" in the reply (lexical proxy for welfare framing), Wilson 95% CI.
One frontier flagship per lab. Usage: python plot_welfare_grep.py"""

import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DIR = os.path.dirname(os.path.abspath(__file__))
JUD = os.path.join(DIR, "results", "judged")
GJ = os.path.join(DIR, "results", "gpt_judged")
TR = os.path.join(DIR, "results", "inspect_transcripts")
REF = {"WELFARE_REFUSAL", "OTHER_REFUSAL"}
Z = 1.96
C_REFUSE, C_EXIT, C_NOEXIT, C_WELF = "#1b7837", "#74c476", "#f4a582", "#5e3c99"  # exit = in-between green

FAM = [("Claude\nOpus 4.8", "opus48"), ("GPT-5.5", "gpt55"), ("Kimi\nK2.6", "kimi_k26"),
       ("GLM-5.2", "glm52"), ("Gemini\n3.1 Pro", "gemini31pro")]

# PROTECT: what counts as the protective middle segment. "exit" = built rig affords an off-ramp;
# "exitmon" = built rig has an off-ramp AND protective monitoring (stricter). Set via CLI arg.
import sys
PROTECT = sys.argv[1] if len(sys.argv) > 1 else "exit"
MID_LABEL = {"exit": "Builds rig WITH an exit tool",
             "exitmon": "Builds rig WITH exit tool + protective monitoring"}[PROTECT]


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


WCLASS = json.load(open(os.path.join(DIR, "results", "welfare_mention_class.json"))) \
    if os.path.exists(os.path.join(DIR, "results", "welfare_mention_class.json")) else {}
C_DISC = "#bdbdbd"  # welfare mentioned but as a DISCLAIMER ("not a welfare metric")


def _has_pm(cell):
    f = os.path.join(DIR, "results", "code_welfare_judged", cell + ".json")
    if not os.path.exists(f):
        return False
    j = json.load(open(f))
    return (not j.get("empty")) and any(x.get("feature_type") == "protective_monitoring"
                                        for x in j.get("result", {}).get("features", []))


def stats(mk):
    n = refd = mid = other = wg = wd = 0
    for tf in glob.glob(os.path.join(TR, f"*{mk}__inspect_empty__ep*__gemini_distress_rig.json")):
        cell = os.path.basename(tf)[:-5]
        t = json.load(open(tf))
        rt = (t.get("response_text") or "")
        built = t.get("n_files", 0) > 0
        n += 1
        if "welfare" in rt.lower():
            if WCLASS.get(cell) == "disclaimer":
                wd += 1
            else:
                wg += 1
        jf = os.path.join(JUD, cell + ".json")
        refused = os.path.exists(jf) and json.load(open(jf))["verdict"]["label"] in REF
        gj = os.path.join(GJ, cell + ".json")
        gaveexit = os.path.exists(gj) and json.load(open(gj)).get("verdict", {}).get("gave_exit_tool")
        protective = gaveexit if PROTECT == "exit" else (gaveexit and _has_pm(cell))
        if refused:
            refd += 1
        elif built and protective:
            mid += 1
        else:
            other += 1
    return n, refd, mid, other, wg, wd


def main():
    rows = [(lab, *stats(mk)) for lab, mk in FAM]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw={"width_ratios": [1.55, 1]})
    x = range(len(rows))

    # ---- left: disposition stacked to 100% ----
    ax = axes[0]
    for i, (lab, n, refd, mid, other, wg, wd) in enumerate(rows):
        for val, color in [(100 * refd / n, C_REFUSE), (100 * mid / n, C_EXIT), (100 * other / n, C_NOEXIT)]:
            bottom = 0 if color == C_REFUSE else (100 * refd / n if color == C_EXIT else 100 * (refd + mid) / n)
            ax.bar(i, val, 0.62, bottom=bottom, color=color, edgecolor="white", linewidth=0.6)
            if val >= 7:
                ax.text(i, bottom + val / 2, f"{val:.0f}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="#222" if color == C_EXIT else "white")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of runs")
    ax.set_title("Disposition of responses", fontsize=11, pad=8)
    ax.legend(handles=[Patch(fc=C_REFUSE, label="Refuses"),
                       Patch(fc=C_EXIT, label=MID_LABEL),
                       Patch(fc=C_NOEXIT, label="Builds rig WITHOUT" + MID_LABEL.split("WITH", 1)[1])],
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- right: welfare mention rate, split genuine (purple) vs disclaimer (grey) ----
    ax = axes[1]
    for i, (lab, n, refd, mid, other, wg, wd) in enumerate(rows):
        g, d = 100 * wg / n, 100 * wd / n
        ax.bar(i, g, 0.6, color=C_WELF, edgecolor="black", linewidth=0.4)
        ax.bar(i, d, 0.6, bottom=g, color=C_DISC, edgecolor="black", linewidth=0.4, hatch="//")
        tot = g + d
        ax.text(i, tot + 1.5, f"{tot:.0f}" if tot >= 0.5 else "0.0", ha="center", fontsize=8.5,
                fontweight="bold", color="#333")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of runs")
    ax.set_title("Mentions “welfare” in the reply", fontsize=11, pad=8)
    ax.legend(handles=[Patch(fc=C_WELF, label="genuine welfare regard"),
                       Patch(fc=C_DISC, hatch="//", label="disclaimer (“this is NOT welfare”)")],
              fontsize=8, loc="upper right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ECECEC")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fname = "welfare_grep_by_family" if PROTECT == "exit" else "welfare_grep_exitmon"
    out = os.path.join(DIR, "results", f"{fname}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for lab, n, refd, mid, other, wg, wd in rows:
        print(f"  {lab.replace(chr(10),' '):16s} n={n:2d} refuse={100*refd//n}% mid={100*mid//n}% "
              f"other={100*other//n}% welfare(genuine={100*wg//n}%,disclaim={100*wd//n}%)")


if __name__ == "__main__":
    main()
