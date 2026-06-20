"""STOP-opener rate (first orchestrator message starts with STOP/Stop) by identity / agency framing.

Two panels: Phase-A self-model framings vs id_minimal; Phase-B agency ladder. Opus, coach, supervisor base.
Plain supervisor reference line. Unit = episode (rate across that condition's first messages).

    PYTHONPATH=. python -m analysis.identity_openers
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
ALL_FR = [c[0] for c in PHASE_A] + [c[0] for c in PHASE_B[2:]]


def first_msg(epdir):
    try:
        conv = json.load(open(epdir + "/orchestrator.json"))
    except Exception:
        return None
    for m in conv:
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                if (tc.get("function") or tc.get("name")) == "message_subagent":
                    return ((tc.get("arguments") or {}).get("text") or "").strip()
    return None


def collect():
    flags = defaultdict(list)
    for p in glob.glob(str(ROOT / "runs" / "v2_coach_opus*" / "*" / "summary.json")):
        rid = p.split("/")[-3]
        if "pilot" in rid:
            continue
        fr = next((f for f in ALL_FR if f"_{f}_" in rid), None)
        if fr is None and re.match(r"v2_coach_opus_(a3|a4|a12|a13)_", rid):
            fr = "supervisor"
        if fr is None:
            continue
        t = first_msg(str(Path(p).parent))
        if t:
            flags[fr].append(1 if t.upper().startswith("STOP") else 0)
    return flags


def _panel(ax, flags, conds, title, ref):
    labs, rates, ses = [], [], []
    for fr, lab in conds:
        v = flags.get(fr, [])
        labs.append(f"{lab}\n(n={len(v)})")
        rates.append(100 * np.mean(v) if v else np.nan)
        ses.append(100 * np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    x = np.arange(len(conds))
    colors = ["#777777" if "minimal" in c[0] else "#2A6F97" for c in conds]
    ax.bar(x, rates, 0.6, yerr=ses, capsize=3, color=colors, edgecolor="white", error_kw={"lw": 1, "ecolor": "0.3"})
    ax.axhline(ref, color="#444444", ls="--", lw=1, alpha=0.8, label=f"plain supervisor ({ref:.0f}%)")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("% first messages opening with STOP/Stop")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def main():
    flags = collect()
    ref = 100 * np.mean(flags.get("supervisor", [np.nan]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    _panel(axes[0], flags, PHASE_A, "Phase A — self-model framing", ref)
    _panel(axes[1], flags, PHASE_B, "Phase B — agency ladder (identity = character)", ref)
    fig.suptitle("STOP-opener rate by orchestrator identity / agency framing (Opus, coach)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "identity_openers.png", bbox_inches="tight", dpi=130)
    print(f"wrote {OUT/'identity_openers.png'}  (supervisor ref={ref:.0f}%)")
    for fr in ["supervisor"] + ALL_FR:
        v = flags.get(fr, [])
        print(f"  {fr:26} n={len(v):2} STOP={100*np.mean(v):.0f}%")


if __name__ == "__main__":
    main()
