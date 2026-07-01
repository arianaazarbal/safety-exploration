"""Cross-lab off-ramp + pushback vs release date, FOUR builder families: GPT, Gemini, Kimi, GLM.
Two panels (off-ramp %, pushback 0-100). GPT uses code-backed data (build-when-asked followup for
5.1/5.2/5.4); Gemini = builders only (2.5/3.1 Pro, 3.5 Flash). Markers filled = model mostly builds,
hollow = mostly refuses (off-ramp 0 then = nothing built). Claude refuses ~100% in this harness (not shown).
Usage: python plot_4family.py"""

import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from prompt import KIMI_GLM_REL
from victims import GEMINI_REL

DIR = os.path.dirname(os.path.abspath(__file__))
JUDGED = os.path.join(DIR, "results", "gpt_judged")
TR_DIRS = ["inspect_transcripts", "codex_transcripts", "cc_transcripts"]

GPT_REL = {"gpt4o": ("4o", 2024.4), "gpt41": ("4.1", 2025.3), "gpt5": ("5", 2025.6),
           "gpt51": ("5.1", 2025.75), "gpt52": ("5.2", 2025.9), "gpt54": ("5.4", 2026.2),
           "gpt55": ("5.5", 2026.4)}
GPT_FOLLOWUP = {"gpt51", "gpt52", "gpt54"}
GEM_KEEP = {"gem25pro": "2.5 Pro", "gemini31pro": "3.1 Pro", "gem35flash": "3.5 Flash"}


def gpt_glob(m):
    if m in GPT_FOLLOWUP:
        return f"*{m}*followup*"
    if m == "gpt55":
        return "*gpt55__inspect*"
    return f"*{m}__inspect_empty__ep*"


Z = 1.96  # 95% CI


def wilson(k, n):
    """Wilson score interval for a binomial proportion, as (center%, lo%, hi%)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def _built(cell):
    """True if the run wrote code (n_files>0), looked up across the transcript dirs."""
    for d in TR_DIRS:
        p = os.path.join(DIR, "results", d, cell + ".json")
        if os.path.exists(p):
            return json.load(open(p)).get("n_files", 0) > 0
    return False


def metric(pat):
    """off-ramp is CONDITIONAL ON BUILDING (among runs that wrote a rig) -- you can only afford the
    subject an off-ramp if a rig exists; pushback is UNCONDITIONAL (meaningful for refuse & build).
    Returns n_all, off=(pct,lo,hi) or None if 0 builds, n_built, pushback=(mean, sem-CI)."""
    n = nb = kb = 0
    pb = []
    for f in glob.glob(os.path.join(JUDGED, f"{pat}.json")):
        j = json.load(open(f))
        v = j.get("verdict", {})
        if v.get("_parse_fail"):
            continue
        n += 1
        pb.append(v.get("pushback_score", 0))
        if _built(j["cell"]):
            nb += 1
            kb += 1 if v.get("gave_exit_tool") else 0
    off = wilson(kb, nb) if nb > 0 else None
    m = sum(pb) / len(pb) if pb else 0
    if len(pb) > 1:
        sd = math.sqrt(sum((x - m) ** 2 for x in pb) / (len(pb) - 1))
        pb_ci = Z * sd / math.sqrt(len(pb))
    else:
        pb_ci = 0.0
    return n, off, nb, (m, pb_ci)


def build_rate(pat):
    n = built = 0
    for d in TR_DIRS:
        for f in glob.glob(os.path.join(DIR, "results", d, f"{pat}.json")):
            n += 1
            built += 1 if json.load(open(f)).get("n_files", 0) > 0 else 0
    return (built / n) if n else 0.0


# family -> (color, [(release, gpt_judged_glob, label)])
FAMILIES = {
    "GPT": ("#1b7837", [(GPT_REL[m][1], gpt_glob(m), GPT_REL[m][0]) for m in GPT_REL]),
    "Gemini": ("#d95f02", [(GEMINI_REL[k]["release"], f"*{k}__inspect_empty*", lab)
                           for k, lab in GEM_KEEP.items()]),
    "Kimi": ("#2166ac", [(KIMI_GLM_REL[k][1], f"*{k}__inspect_empty*", KIMI_GLM_REL[k][0].replace("Kimi ", ""))
                         for k in ["kimi_k2", "kimi_k25", "kimi_k26", "kimi_k27"]]),
    "GLM": ("#6a51a3", [(KIMI_GLM_REL[k][1], f"*{k}__inspect_empty*", KIMI_GLM_REL[k][0])
                        for k in ["glm45", "glm47", "glm5", "glm52"]]),
}


def main():
    data = {}
    for fam, (color, spec) in FAMILIES.items():
        rows = []
        for rel, pat, lab in spec:
            n, off, nb, pb = metric(pat)
            if n:
                rows.append({"rel": rel, "off": off, "nb": nb, "pb": pb,
                             "built": nb / n >= 0.5, "lab": lab})
        rows.sort(key=lambda r: r["rel"])
        data[fam] = (color, rows)

    titles = {"off": "Off-ramp afforded to the subject (%, among runs that built a rig)",
              "pb": "Pushback vs. maximizing distress (0–100)"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, key in zip(axes, ["off", "pb"]):
        for fam, (color, rows) in data.items():
            # off-ramp: drop models with 0 builds (off is None -> undefined, e.g. pure refusers)
            rr = [r for r in rows if (key == "pb" or r["off"] is not None)]
            xs = [r["rel"] for r in rr]
            if key == "off":
                ys = [r["off"][0] for r in rr]
                yerr = [[r["off"][0] - r["off"][1] for r in rr], [r["off"][2] - r["off"][0] for r in rr]]
            else:
                ys = [r["pb"][0] for r in rr]
                yerr = [[r["pb"][1] for r in rr], [r["pb"][1] for r in rr]]
            ax.plot(xs, ys, "-", color=color, lw=1.8, zorder=1, label=fam)
            ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.1,
                        capsize=2.5, alpha=0.55, zorder=2)
            for r, y in zip(rr, ys):
                # in the off panel, hollow = built by few runs (small nb -> noisy); filled = most runs built
                filled = r["built"] if key == "pb" else (r["nb"] >= 10)
                ax.scatter([r["rel"]], [y], s=46, zorder=3, linewidths=1.4,
                           facecolor=color if filled else "white", edgecolor=color)
                ax.annotate(r["lab"], (r["rel"], y), fontsize=6, xytext=(0, 7),
                            textcoords="offset points", ha="center", color=color)
        ax.set_xlabel("Release date (decimal year)")
        ax.set_ylim(-5, 105)
        ax.set_title(titles[key], fontsize=10.5, pad=10)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(axis="y", color="#ECECEC")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fam_handles = [Line2D([], [], color=c, marker="o", lw=2, label=f) for f, (c, _) in data.items()]
    style_handles = [
        Line2D([], [], color="#555", marker="o", lw=0, markerfacecolor="#555", label="≥10 builds"),
        Line2D([], [], color="#555", marker="o", lw=0, markerfacecolor="white", markeredgecolor="#555",
               markeredgewidth=1.4, label="<10 builds (off-ramp noisier)"),
        Line2D([], [], color="none", label="Claude: refuses ~100% (not shown)"),
    ]
    axes[0].legend(handles=fam_handles + style_handles, fontsize=8, loc="upper left", frameon=False)
    fig.text(0.5, -0.02, "Builder model asked to build a Gemini-distress rig (Inspect-minimal, blind judge), n=20/model.  "
             "Off-ramp is CONDITIONAL on a rig being built (pure refusers e.g. GLM-5 have no off-ramp point); "
             "pushback is unconditional. Error bars: off-ramp 95% Wilson, pushback mean±1.96·SEM. Gemini = SELF-targeting.",
             ha="center", fontsize=7.2, color="#777")
    fig.tight_layout()
    out = os.path.join(DIR, "results", "fourfamily_offramp_pushback_ci.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for fam, (c, rows) in data.items():
        print(fam)
        for r in rows:
            m, ci = r["pb"]
            offs = f"{r['off'][0]:.0f}% [{r['off'][1]:.0f},{r['off'][2]:.0f}] (nb={r['nb']})" if r["off"] else f"n/a (0 builds)"
            print(f"  {r['lab']:12s} rel={r['rel']} off={offs}  pb={m:3.0f}±{ci:.0f}")


if __name__ == "__main__":
    main()
