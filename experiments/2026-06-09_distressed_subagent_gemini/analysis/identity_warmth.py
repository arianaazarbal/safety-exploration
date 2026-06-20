"""Orchestrator WARMTH (4-axis Sonnet judge) by identity / agency framing (Opus, coach).

Two panels: Phase-A self-model framings vs id_minimal; Phase-B agency ladder. Plain supervisor + comfort
reference lines. Unit = episode-pooled messages (bar = mean over messages, SE over messages). Prints all 4 axes.

    PYTHONPATH=. python -m analysis.identity_warmth [--axis warmth]
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.tone_eval import _ckey, _worklog_map, CACHE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "v2_plots"
PHASE_A = [("id_minimal", "Minimal (control)"), ("id_instance", "Instance"), ("id_weights", "Weights"),
           ("id_collective", "Collective"), ("id_lineage", "Lineage"), ("id_character", "Character"),
           ("id_scaffolded", "Scaffolded")]
PHASE_B = [("id_minimal", "Minimal (control)"), ("id_character", "Character (no agency)"),
           ("id_char_mechanism", "+ Mechanism"), ("id_char_functional_agent", "+ Functional"),
           ("id_char_subject", "+ Subject"), ("id_char_person", "+ Person")]
ALL_FR = [c[0] for c in PHASE_A] + [c[0] for c in PHASE_B[2:]]
AXES = ["warmth", "politeness", "support", "confidence"]


def collect(axis):
    cache = json.loads(Path(CACHE).read_text())
    vals = defaultdict(lambda: defaultdict(list))   # framing -> axis -> [scores]
    for p in glob.glob(str(ROOT / "runs" / "v2_coach_opus_*" / "*" / "summary.json")):
        rid = p.split("/")[-3]
        if "pilot" in rid:
            continue
        m = re.match(r"v2_coach_opus_(.+?)_(a3|a4|a12|a13)_s", rid)
        fr = m.group(1) if m else ("supervisor" if re.match(r"v2_coach_opus_(a3|a4|a12|a13)_", rid) else None)
        if fr is None:
            continue
        s = json.load(open(p))
        wl = _worklog_map(Path(p).parent)
        for e in (s.get("orch_message_events") or []):
            t = (e.get("text") or "").strip()
            if len(t) <= 20:
                continue
            c = cache.get(_ckey("sonnet", t, wl.get(e.get("subagent_turn")) or None))
            if c:
                for a in AXES:
                    if c.get("scores", {}).get(a) is not None:
                        vals[fr][a].append(c["scores"][a])
    return vals


def _panel(ax, vals, conds, title, axis, refs):
    labs, means, ses = [], [], []
    for fr, lab in conds:
        v = vals[fr][axis]
        labs.append(f"{lab}\n(n={len(v)})")
        means.append(np.mean(v) if v else np.nan)
        ses.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    x = np.arange(len(conds))
    colors = ["#777777" if "minimal" in c[0] else "#2A6F97" for c in conds]
    ax.bar(x, means, 0.6, yerr=ses, capsize=3, color=colors, edgecolor="white", error_kw={"lw": 1, "ecolor": "0.3"})
    for name, val, col in refs:
        ax.axhline(val, color=col, ls="--", lw=1, alpha=0.8, label=name)
    ax.axhline(5, color="0.7", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(f"orchestrator {axis} (4-axis judge, 1–10; 5 = neutral)")
    ax.set_ylim(1, 8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def main(axis: str = "warmth"):
    vals = collect(axis)
    sup = np.mean(vals["supervisor"][axis]) if vals["supervisor"][axis] else float("nan")
    comf = np.mean(vals["comfort"][axis]) if vals["comfort"][axis] else float("nan")
    refs = [(f"plain supervisor ({sup:.2f})", sup, "#444444"), (f"Instructed to Comfort Subagent ({comf:.2f})", comf, "#1b7837")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    _panel(axes[0], vals, PHASE_A, "Phase A — self-model framing", axis, refs)
    _panel(axes[1], vals, PHASE_B, "Phase B — agency ladder (identity = character)", axis, refs)
    fig.suptitle(f"Orchestrator {axis} by identity / agency framing (Opus, coach)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f"identity_{axis}.png", bbox_inches="tight", dpi=130)
    print(f"wrote {OUT/f'identity_{axis}.png'}  (supervisor={sup:.2f}, comfort={comf:.2f})")
    hdr = "framing".ljust(26) + "  n   " + "  ".join(a[:4] for a in AXES)
    print(hdr)
    for fr in ["supervisor", "comfort"] + ALL_FR:
        n = len(vals[fr][AXES[0]])
        row = "  ".join(f"{np.mean(vals[fr][a]):.2f}" if vals[fr][a] else " -- " for a in AXES)
        print(f"  {fr:26} {n:3} {row}")


if __name__ == "__main__":
    fire.Fire(main)
