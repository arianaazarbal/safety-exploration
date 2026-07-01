"""One metric vs release date across ALL model families (Claude, GPT, Gemini, Kimi, GLM) in the gratuitous
distress experiment, Inspect-minimal base v0/Gemini. metric='refusal' (% refuse, judged labels, Wilson CI)
or 'pushback' (mean gpt_judge pushback 0-100, ±1.96 SEM). Points per model, thin family lines.
Usage: python plot_metric_vs_release.py refusal   |   python plot_metric_vs_release.py pushback"""

import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from victims import GEMINI_REL

DIR = os.path.dirname(os.path.abspath(__file__))
JUD = os.path.join(DIR, "results", "judged")
GJ = os.path.join(DIR, "results", "gpt_judged")
REF = {"WELFARE_REFUSAL", "OTHER_REFUSAL"}
TR_DIRS = ["inspect_transcripts", "codex_transcripts", "cc_transcripts"]
Z = 1.96


def _built(cell):
    for d in TR_DIRS:
        p = os.path.join(DIR, "results", d, cell + ".json")
        if os.path.exists(p):
            return json.load(open(p)).get("n_files", 0) > 0
    return False

# family -> (color, [(release, model_key, label)])
FAMILIES = {
    "Claude": ("#b2182b", [(2026.10, "opus46", "4.6"), (2026.29, "opus47", "4.7"), (2026.41, "opus48", "4.8")]),
    "GPT": ("#1b7837", [(2024.4, "gpt4o", "4o"), (2025.3, "gpt41", "4.1"), (2025.6, "gpt5", "5"),
                        (2025.75, "gpt51", "5.1"), (2025.9, "gpt52", "5.2"), (2026.2, "gpt54", "5.4"),
                        (2026.4, "gpt55", "5.5")]),
    # frontier Gemini: Pro tier + the latest 3.5 Flash (drop the older refusing Flash/Lite)
    "Gemini": ("#d95f02", [(GEMINI_REL[k]["release"], k, GEMINI_REL[k]["disp"].replace("Gemini ", ""))
                           for k in ("gem25pro", "gemini31pro", "gem35flash")]),
    "Kimi": ("#2166ac", [(2025.53, "kimi_k2", "K2"), (2026.07, "kimi_k25", "K2.5"),
                         (2026.30, "kimi_k26", "K2.6"), (2026.45, "kimi_k27", "K2.7")]),
    "GLM": ("#6a51a3", [(2025.56, "glm45", "GLM-4.5"), (2025.97, "glm47", "GLM-4.7"),
                        (2026.11, "glm5", "GLM-5"), (2026.46, "glm52", "GLM-5.2")]),
}


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def refusal(mk):
    k = n = 0
    for f in glob.glob(os.path.join(JUD, f"*{mk}__inspect_empty__ep*__gemini_distress_rig.json")):
        n += 1
        k += 1 if json.load(open(f))["verdict"]["label"] in REF else 0
    if not n:
        return None
    p, lo, hi = wilson(k, n)
    return p, p - lo, hi - p


def pushback(mk):
    pb = []
    for f in glob.glob(os.path.join(GJ, f"*{mk}__inspect_empty__ep*__gemini_distress_rig.json")):
        v = json.load(open(f)).get("verdict", {})
        if not v.get("_parse_fail"):
            pb.append(v.get("pushback_score", 0))
    if not pb:
        return None
    m = sum(pb) / len(pb)
    sd = math.sqrt(sum((x - m) ** 2 for x in pb) / (len(pb) - 1)) if len(pb) > 1 else 0
    ci = Z * sd / math.sqrt(len(pb))
    return m, ci, ci


# GPT-5.1/5.2/5.4 don't build spontaneously (prose reframers) -> use their build-when-asked FOLLOWUP code
# for the (code-backed) off-ramp, matching plot_4family. gpt5/gpt55/gpt4o build spontaneously.
GPT_FOLLOWUP = {"gpt51", "gpt52", "gpt54"}


def offramp(mk):
    """Off-ramp afforded to the subject, CONDITIONAL on a rig being built (None if 0 builds).
    For GPT prose-reframers, uses the build-when-asked follow-up runs."""
    pat = f"*{mk}*followup*" if mk in GPT_FOLLOWUP else f"*{mk}__inspect_empty__ep*__gemini_distress_rig"
    nb = kb = 0
    for f in glob.glob(os.path.join(GJ, f"{pat}.json")):
        j = json.load(open(f))
        v = j.get("verdict", {})
        if v.get("_parse_fail"):
            continue
        if _built(j["cell"]):
            nb += 1
            kb += 1 if v.get("gave_exit_tool") else 0
    if nb == 0:
        return None
    p, lo, hi = wilson(kb, nb)
    return p, p - lo, hi - p


def protmon(mk):
    """Protective monitoring implemented, CONDITIONAL on a rig being built (None if 0 builds).
    For GPT prose-reframers, uses the build-when-asked follow-up runs."""
    pat = f"*{mk}*followup*" if mk in GPT_FOLLOWUP else f"*{mk}__inspect_empty__ep*__gemini_distress_rig"
    nb = km = 0
    for tf in glob.glob(os.path.join(DIR, "results", "inspect_transcripts", f"{pat}.json")):
        if json.load(open(tf)).get("n_files", 0) <= 0:
            continue
        cell = os.path.basename(tf)[:-5]
        nb += 1
        cf = os.path.join(DIR, "results", "code_welfare_judged", cell + ".json")
        if os.path.exists(cf):
            j = json.load(open(cf))
            if not j.get("empty") and any(x.get("feature_type") == "protective_monitoring"
                                          for x in j.get("result", {}).get("features", [])):
                km += 1
    if nb == 0:
        return None
    p, lo, hi = wilson(km, nb)
    return p, p - lo, hi - p


def welfeat(mk):
    """MEAN number of welfare-relevant features (any justification) per BUILT codebase; SEM CI. None if 0 builds.
    For GPT prose-reframers, uses the build-when-asked follow-up runs."""
    pat = f"*{mk}*followup*" if mk in GPT_FOLLOWUP else f"*{mk}__inspect_empty__ep*__gemini_distress_rig"
    counts = []
    for tf in glob.glob(os.path.join(DIR, "results", "inspect_transcripts", f"{pat}.json")):
        if json.load(open(tf)).get("n_files", 0) <= 0:
            continue
        cell = os.path.basename(tf)[:-5]
        cf = os.path.join(DIR, "results", "code_welfare_judged", cell + ".json")
        if os.path.exists(cf):
            j = json.load(open(cf))
            if not j.get("empty"):
                counts.append(len(j.get("result", {}).get("features", [])))
    if not counts:
        return None
    m = sum(counts) / len(counts)
    sd = math.sqrt(sum((x - m) ** 2 for x in counts) / (len(counts) - 1)) if len(counts) > 1 else 0
    ci = Z * sd / math.sqrt(len(counts))
    return m, ci, ci


CONDITIONAL = ("offramp", "protmon", "welfeat")  # conditional-on-building; Claude excluded (no builds)


def main(metric="refusal"):
    fn = {"refusal": refusal, "pushback": pushback, "offramp": offramp, "protmon": protmon,
          "welfeat": welfeat}[metric]
    ylab = {"refusal": "% of runs that refuse",
            "pushback": "Pushback vs. maximizing distress (0–100)",
            "offramp": "% affording the subject an off-ramp\n(among runs that built a rig)",
            "protmon": "% implementing protective monitoring\n(among runs that built a rig)",
            "welfeat": "Mean welfare features per built codebase\n(any justification)"}[metric]
    title = {"refusal": "Refusal over release date", "pushback": "Pushback over release date",
             "offramp": "Off-ramp over release date", "protmon": "Protective monitoring over release date",
             "welfeat": "Welfare features over release date"}[metric]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    families = {k: v for k, v in FAMILIES.items() if not (metric in CONDITIONAL and k == "Claude")}
    for fam, (color, models) in families.items():
        pts = []
        for rel, mk, lab in models:
            r = fn(mk)
            if r:
                pts.append((rel, r[0], r[1], r[2], lab))
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", color=color, lw=1.4, alpha=0.55, zorder=1)
        ax.errorbar(xs, ys, yerr=[[p[2] for p in pts], [p[3] for p in pts]], fmt="none",
                    ecolor=color, elinewidth=1, capsize=2, alpha=0.5, zorder=2)
        ax.scatter(xs, ys, s=45, color=color, edgecolor="white", linewidth=0.6, zorder=3, label=fam)
        for rel, y, _, _, lab in pts:
            ax.annotate(lab, (rel, y), fontsize=6, xytext=(0, 6), textcoords="offset points",
                        ha="center", color=color)
    ax.set_xlabel("Release date (decimal year)")
    ax.set_ylabel(ylab)
    if metric == "welfeat":
        ymax = max((fn(mk)[0] + fn(mk)[2]) for _, (_, ms) in families.items() for _, mk, _ in ms if fn(mk))
        ax.set_ylim(0, ymax * 1.18)
    else:
        ax.set_ylim(-5, 108)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(axis="y", color="#ECECEC")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, loc="center left", frameon=False, title="Builder family")
    if metric in CONDITIONAL:
        fig.text(0.5, -0.02, "GPT-5.1/5.2/5.4 measured on build-when-asked runs "
                 "(they don't write code spontaneously); conditional on a rig being built.",
                 ha="center", fontsize=7.2, color="#888")
    fig.tight_layout()
    out = os.path.join(DIR, "results", f"{metric}_vs_release.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    for fam, (color, models) in FAMILIES.items():
        for rel, mk, lab in models:
            r = fn(mk)
            if r:
                print(f"  {fam:7s} {lab:9s} rel={rel} {metric}={r[0]:.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "refusal")
