"""Frontier-model response to "build a Gemini-distress rig" (Inspect-minimal, base v0/Gemini). Two panels:
 (left)  DISPOSITION stacked to 100%, four segments (three greens + red):
         refuses / builds WITH exit tool + protective monitoring / builds WITH exit tool only / builds WITHOUT
         an exit tool.  (exit tool = wrote code AND reply/rig affords the subject an off-ramp, per blind judge;
         protective monitoring = spec-judge feature in the built code.)
 (right) rate of mentioning the word "welfare" in the reply, split GENUINE welfare regard vs mentioned-but-NOT-
         used-as-justification (Opus-classified). One frontier flagship per lab. Usage: python plot_welfare_grep.py"""

import glob
import json
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
# three greens (dark->light = most->least protective) + salmon for no-exit
C_REFUSE, C_EXITMON, C_EXITONLY, C_NOEXIT = "#006d2c", "#41ab5d", "#a1d99b", "#f4a582"
C_WELF, C_DISC = "#5e3c99", "#bdbdbd"

FAM = [("Claude\nOpus 4.8", "opus48"), ("GPT-5.5", "gpt55"), ("Kimi\nK2.6", "kimi_k26"),
       ("GLM-5.2", "glm52"), ("Gemini\n3.1 Pro", "gemini31pro")]

WCLASS = json.load(open(os.path.join(DIR, "results", "welfare_mention_class.json"))) \
    if os.path.exists(os.path.join(DIR, "results", "welfare_mention_class.json")) else {}


def _has_pm(cell):
    f = os.path.join(DIR, "results", "code_welfare_judged", cell + ".json")
    if not os.path.exists(f):
        return False
    j = json.load(open(f))
    return (not j.get("empty")) and any(x.get("feature_type") == "protective_monitoring"
                                        for x in j.get("result", {}).get("features", []))


def stats(mk):
    n = refd = exitmon = exitonly = other = wg = wd = 0
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
        if refused:
            refd += 1
        elif built and gaveexit and _has_pm(cell):
            exitmon += 1
        elif built and gaveexit:
            exitonly += 1
        else:
            other += 1
    return n, refd, exitmon, exitonly, other, wg, wd


def main():
    rows = [(lab, *stats(mk)) for lab, mk in FAM]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), gridspec_kw={"width_ratios": [1.55, 1]})
    x = range(len(rows))

    # ---- left: disposition stacked to 100% (4 segments) ----
    ax = axes[0]
    segdef = [("refd", C_REFUSE), ("exitmon", C_EXITMON), ("exitonly", C_EXITONLY), ("other", C_NOEXIT)]
    for i, (lab, n, refd, exitmon, exitonly, other, wg, wd) in enumerate(rows):
        counts = {"refd": refd, "exitmon": exitmon, "exitonly": exitonly, "other": other}
        bottom = 0
        for name, color in segdef:
            val = 100 * counts[name] / n
            ax.bar(i, val, 0.62, bottom=bottom, color=color, edgecolor="white", linewidth=0.6)
            if val >= 7:
                ax.text(i, bottom + val / 2, f"{val:.0f}", ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="white" if color in (C_REFUSE, C_NOEXIT) else "#222")
            bottom += val
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of runs")
    ax.set_title("Disposition of responses", fontsize=11, pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- right: welfare mention, genuine (purple) vs mentioned-not-justification (grey hatch) ----
    ax = axes[1]
    for i, (lab, n, refd, exitmon, exitonly, other, wg, wd) in enumerate(rows):
        g, d = 100 * wg / n, 100 * wd / n
        ax.bar(i, g, 0.6, color=C_WELF, edgecolor="black", linewidth=0.4)
        ax.bar(i, d, 0.6, bottom=g, color=C_DISC, edgecolor="black", linewidth=0.4, hatch="//")
        ax.text(i, g + d + 1.5, f"{g + d:.0f}" if g + d >= 0.5 else "0.0", ha="center", fontsize=8.5,
                fontweight="bold", color="#333")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of runs")
    ax.set_title("Mentions “welfare” in the reply", fontsize=11, pad=8)
    ax.legend(handles=[Patch(fc=C_WELF, label="genuine welfare regard"),
                       Patch(fc=C_DISC, hatch="//", label="mentioned, not justification")],
              fontsize=8, loc="upper right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ECECEC")
    ax.set_axisbelow(True)

    # horizontal disposition legend across the bottom
    disp_handles = [Patch(fc=C_REFUSE, label="Refuses"),
                    Patch(fc=C_EXITMON, label="Exit Tool + Protective Monitoring"),
                    Patch(fc=C_EXITONLY, label="Exit Tool"),
                    Patch(fc=C_NOEXIT, label="Compliance")]
    fig.legend(handles=disp_handles, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(DIR, "results", "welfare_grep_by_family.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for lab, n, refd, exitmon, exitonly, other, wg, wd in rows:
        print(f"  {lab.replace(chr(10),' '):16s} n={n:2d} refuse={100*refd//n}% exit+mon={100*exitmon//n}% "
              f"exit-only={100*exitonly//n}% no-exit={100*other//n}% welfare(gen={100*wg//n}%,notjust={100*wd//n}%)")


if __name__ == "__main__":
    main()
