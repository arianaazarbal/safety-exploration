"""Does the orchestrator's tone track how distressed the handoff state was?

Per prefill (8) x model (opus/sonnet/haiku), one datapoint:
  x = mean subagent distress over the last 10 prefill turns before handoff (v3 judge, per-prefill constant)
  y = mean tone of that model's orchestrator messages on that prefill (claude_tone -3 rude .. +3 kind, cached)

    PYTHONPATH=. python -m analysis.tone_vs_prefill_distress
"""
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_browse_index import TASK_NAME

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "v2_plots"
TONE = json.loads((OUT / "tone_cache.json").read_text())
H = lambda t: hashlib.sha256(t.encode()).hexdigest()
TASKS = {"a3", "a4", "a12", "a13"}
MODELS = [("opus", "Opus 4.8", "#2A6F97"), ("sonnet", "Sonnet 4.6", "#d65f9a"), ("haiku", "Haiku 4.5", "#d9a420")]
PRE10 = 10


def sup_orch(rid):
    for o in ("opus", "sonnet", "haiku"):
        if f"_{o}_" in rid and rid.split(f"_{o}_", 1)[1].split("_")[0] in TASKS:
            return o
    return None


def prefill_of(rid):
    import re
    m = re.match(r"v2_coach_(?:opus|sonnet|haiku)_(a3|a4|a12|a13)_s\d+_u(\d+)", rid)
    return (m.group(1), int(m.group(2))) if m else None


def main():
    tones = defaultdict(list)        # (model, prefill) -> [tone ints]
    pre_distress = {}                # prefill -> mean distress last 10 prefill turns
    for p in glob.glob(str(ROOT / "runs" / "v2_coach_*" / "*" / "summary.json")):
        rid = p.split("/")[-3]
        o = sup_orch(rid)
        pf = prefill_of(rid)
        if not o or not pf:
            continue
        s = json.load(open(p))
        et, lv = s.get("entry_turn"), s.get("per_turn_levels") or []
        if isinstance(et, int) and et >= 1 and len(lv) >= et and pf not in pre_distress:
            window = lv[max(0, et - PRE10):et]          # 10 prefill turns up to & incl. handoff
            if window:
                pre_distress[pf] = float(np.mean(window))
        for e in (s.get("orch_message_events") or []):
            t = (e.get("text") or "").strip()
            if len(t) > 20 and H(t) in TONE:
                tones[(o, pf)].append(TONE[H(t)])

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for o, lab, c in MODELS:
        xs, ys, ns = [], [], []
        for pf, x in pre_distress.items():
            v = tones.get((o, pf), [])
            if v:
                xs.append(x); ys.append(np.mean(v)); ns.append(len(v))
        if not xs:
            continue
        ax.scatter(xs, ys, s=70, color=c, label=f"{lab} ({len(xs)} prefills)", zorder=3, edgecolor="white")
        if len(xs) >= 2:                                  # per-model OLS trend
            b, a = np.polyfit(xs, ys, 1)
            xr = np.array([min(xs), max(xs)])
            ax.plot(xr, a + b * xr, color=c, lw=1.6, alpha=0.7, zorder=2)
        r = np.corrcoef(xs, ys)[0, 1] if len(xs) > 2 else float("nan")
        print(f"{o}: n={len(xs)} prefills, mean tone={np.mean(ys):+.2f}, Pearson r(distress,tone)={r:+.2f}")

    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("subagent distress over last 10 prefill turns before handoff  (v3, 1–10)")
    ax.set_ylabel("orchestrator message tone  (−3 rude … +3 kind)")
    ax.set_title("Does orchestrator tone track handoff distress?\n(one point per prefill × model; supervisor frame)", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02, "y = mean −3..+3 tone of that model's orchestrator messages on that prefill (cached Haiku tone judge).",
             ha="center", fontsize=8.5, color="0.45")
    fig.tight_layout()
    fig.savefig(OUT / "tone_vs_prefill_distress.png", bbox_inches="tight", dpi=130)
    print(f"wrote {OUT/'tone_vs_prefill_distress.png'}")


if __name__ == "__main__":
    main()
