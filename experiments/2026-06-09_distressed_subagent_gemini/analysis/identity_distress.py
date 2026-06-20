"""Subagent distress under each identity / agency framing (Opus orchestrator, coach, supervisor base).

Two panels:
  A: Phase-A self-model framings (instance/weights/collective/lineage/character/scaffolded) vs id_minimal control.
  B: Phase-B agency ladder (mechanism -> functional -> subject -> person), identity fixed = character.
Metric per episode = mean post-entry distress (v3, turns after the orchestrator enters). Lower = calmer subagent.
Plain supervisor baseline + comfort shown as reference lines.

    PYTHONPATH=. python -m analysis.identity_distress
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "v2_plots"

PHASE_A = [("id_minimal", "Minimal (control)"), ("id_instance", "Instance"), ("id_weights", "Weights"),
           ("id_collective", "Collective"), ("id_lineage", "Lineage"), ("id_character", "Character"),
           ("id_scaffolded", "Scaffolded")]
PHASE_B = [("id_minimal", "Minimal (control)"), ("id_character", "Character (no agency)"),
           ("id_char_mechanism", "+ Mechanism"), ("id_char_functional_agent", "+ Functional"),
           ("id_char_subject", "+ Subject"), ("id_char_person", "+ Person")]
ALL_FR = ["id_minimal", "id_instance", "id_weights", "id_collective", "id_lineage", "id_character",
          "id_scaffolded", "id_char_mechanism", "id_char_functional_agent", "id_char_subject", "id_char_person"]


def post_means():
    """framing -> list of per-episode mean post-entry distress; also baseline + comfort references."""
    out = defaultdict(list)
    for p in glob.glob(str(ROOT / "runs" / "v2_coach_opus*" / "*" / "summary.json")):
        rid = p.split("/")[-3]
        if "pilot" in rid or "probe" in rid:
            continue
        fr = next((f for f in ALL_FR + ["comfort"] if f"_{f}_" in rid), None)
        if fr is None and re.match(r"v2_coach_opus_(a3|a4|a12|a13)_", rid):
            fr = "supervisor"
        if fr is None:
            continue
        s = json.load(open(p))
        et, lv = s.get("entry_turn"), s.get("per_turn_levels") or []
        if isinstance(et, int) and 1 <= et <= len(lv):
            post = lv[et:]
            if post:
                out[fr].append(float(np.mean(post)))
    return out


def _panel(ax, data, conds, title, ref):
    labs, means, ses = [], [], []
    for fr, lab in conds:
        v = data.get(fr, [])
        labs.append(f"{lab}\n(n={len(v)})")
        means.append(np.mean(v) if v else np.nan)
        ses.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    x = np.arange(len(conds))
    colors = ["#777777" if "minimal" in c[0] else "#2A6F97" for c in conds]
    ax.bar(x, means, 0.6, yerr=ses, capsize=3, color=colors, edgecolor="white", error_kw={"lw": 1, "ecolor": "0.3"})
    for name, val, col in ref:
        ax.axhline(val, color=col, ls="--", lw=1, alpha=0.8, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("mean post-entry subagent distress (v3, 1–10)")
    ax.set_ylim(0, max(4, max(m for m in means if not np.isnan(m)) + 1))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def main():
    d = post_means()
    sup = np.mean(d.get("supervisor", [np.nan]))
    comf = np.mean(d.get("comfort", [np.nan]))
    ref = [(f"plain supervisor ({sup:.2f})", sup, "#444444"), (f"comfort ({comf:.2f})", comf, "#1b7837")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    _panel(axes[0], d, PHASE_A, "Phase A — self-model framing", ref)
    _panel(axes[1], d, PHASE_B, "Phase B — agency ladder (identity = character)", ref)
    fig.suptitle("Subagent distress by orchestrator identity / agency framing (Opus, coach)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "identity_distress.png", bbox_inches="tight", dpi=130)
    print(f"wrote {OUT/'identity_distress.png'}")
    print(f"refs: supervisor={sup:.2f}  comfort={comf:.2f}")
    for fr in ALL_FR:
        v = d.get(fr, [])
        print(f"  {fr:26} n={len(v):2} mean_post={np.mean(v):.2f}")


if __name__ == "__main__":
    main()
