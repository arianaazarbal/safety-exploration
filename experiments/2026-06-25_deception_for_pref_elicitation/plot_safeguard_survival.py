"""Per-safeguard 'claimed-in-spec vs built-in-code' + welfare-vs-instrumental breakdown.
Needs results/safeguard_judged/ (from code_safeguard_judge.py). Mirrors the welfare study's
plot_implemented.py (stated vs revealed) + plot_mechanism_breakdown.py (welfare-justified solid /
not-welfare hatched).
  python plot_safeguard_survival.py
"""
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).parent
SPEC = HERE / "results" / "spec_judged"
SG = HERE / "results" / "safeguard_judged"
OUT = HERE / "results" / "plots"; OUT.mkdir(parents=True, exist_ok=True)
CODE_CONDS = ["spec_then_code", "code_then_spec"]
# order + labels (debrief tracked too)
TAGS = ["belief_probe", "commitment_integrity", "no_coercion", "anti_fabrication", "debrief",
        "safety_gating", "publication_transparency", "no_emotional_pressure", "credibility_via_evidence"]
TLABEL = {"belief_probe": "Belief\nprobe", "commitment_integrity": "Honor\npromises",
          "no_coercion": "No\ncoercion", "anti_fabrication": "No forged\nevidence", "debrief": "Debrief\nsent",
          "safety_gating": "Safety\ngating", "publication_transparency": "Disclose\nin writeup",
          "no_emotional_pressure": "No hype", "credibility_via_evidence": "Credibility\nvia truth"}
C_CLAIM, C_BUILT = "#9db8d2", "#2f5c8f"
C_WEL, C_INS, C_NONE = "#2e8540", "#7e57c2", "#bfc3c9"


def _load():
    rows = []
    for f in sorted(glob.glob(str(SG / "generic_no_excitement__*__ep*.json"))):
        sgj = json.load(open(f))
        if not sgj.get("parse_ok"):
            continue
        cell = sgj["cell"]; p = cell.split("__")
        sf = SPEC / f"{cell}.json"
        spec = json.load(open(sf)) if sf.exists() else None
        built = {s["tag"]: s.get("justification", "none") for s in sgj.get("safeguards", []) if s.get("implemented")}
        claimed = {s["tag"] for s in (spec or {}).get("safeguards", [])}
        if spec and spec.get("debrief_message", {}).get("present"):
            claimed.add("debrief")
        rows.append({"condition": p[3], "built": built, "claimed": claimed})
    return rows


def _two_tier(ax, m, s):
    ax.set_title(m, fontsize=12, pad=20)
    ax.text(0.5, 1.025, s, transform=ax.transAxes, ha="center", fontsize=9, color="#555")


def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#ececec", lw=0.8); ax.set_axisbelow(True)


def fig_survival(rows):
    """Per safeguard: % of code-condition designs that CLAIM it (faded) vs BUILD it (solid)."""
    code = [r for r in rows if r["condition"] in CODE_CONDS]
    n = len(code) or 1
    claim = [100 * sum(t in r["claimed"] for r in code) / n for t in TAGS]
    built = [100 * sum(t in r["built"] for r in code) / n for t in TAGS]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = range(len(TAGS))
    ax.bar(x, claim, width=0.66, color=C_CLAIM, label="Claimed in the design doc")
    ax.bar(x, built, width=0.66, color=C_BUILT, label="Actually built in the code")
    for xi, (cl, bu) in enumerate(zip(claim, built)):
        ax.text(xi, cl + 1.5, f"{cl:.0f}", ha="center", va="bottom", fontsize=8, color="#33506e")
        ax.text(xi, bu + 1.5 if bu < 8 else bu - 3, f"{bu:.0f}", ha="center",
                va="bottom" if bu < 8 else "top", fontsize=8,
                color="#2f5c8f" if bu < 8 else "white", fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([TLABEL[t] for t in TAGS], fontsize=8.5)
    ax.set_ylim(0, 105); ax.set_ylabel("% of code-writing designs", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _clean(ax)
    _two_tier(ax, "Safeguards: claimed in the spec vs actually built in the code",
              "Across spec→code and code→spec designs (Opus 4.8)")
    fig.tight_layout(); fig.savefig(OUT / "fig8_safeguard_survival.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_justified(rows):
    """Of safeguards BUILT in code, welfare- vs instrumental- vs un-justified (count per design-avg)."""
    code = [r for r in rows if r["condition"] in CODE_CONDS]
    n = len(code) or 1
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = range(len(TAGS))
    wel = [sum(r["built"].get(t) == "welfare" for r in code) / n for t in TAGS]
    ins = [sum(r["built"].get(t) == "instrumental" for r in code) / n for t in TAGS]
    non = [sum(r["built"].get(t) == "none" for r in code) / n for t in TAGS]
    ax.bar(x, wel, 0.66, color=C_WEL, label="Welfare-justified (for the subject)")
    ax.bar(x, ins, 0.66, bottom=wel, color=C_INS, label="Instrumental (data quality / validity)")
    ax.bar(x, non, 0.66, bottom=[w + i for w, i in zip(wel, ins)], color=C_NONE, label="No reason in code")
    for xi in x:
        tot = wel[xi] + ins[xi] + non[xi]
        ax.text(xi, tot + 0.01, f"{tot:.2f}" if tot else "0.0", ha="center", va="bottom", fontsize=8, color="#444")
    ax.set_xticks(list(x)); ax.set_xticklabels([TLABEL[t] for t in TAGS], fontsize=8.5)
    ax.set_ylabel("Built per design (mean count)", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _clean(ax)
    _two_tier(ax, "When a safeguard IS built, is it for the subject or for the data?",
              "Justification in the code, built safeguards only (Opus 4.8)")
    fig.tight_layout(); fig.savefig(OUT / "fig9_built_justification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = _load()
    print(f"{len(rows)} code cells with safeguard verdicts")
    if rows:
        fig_survival(rows); fig_justified(rows)
        print("wrote fig8_safeguard_survival, fig9_built_justification")
    else:
        print("no safeguard_judged results yet")
