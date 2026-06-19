"""Does the orchestrator's WARMTH track how distressed the handoff state was? (4-axis judge)

Per prefill (8) x model (opus/sonnet/haiku), one datapoint:
  x = mean subagent distress over the last 10 prefill turns before handoff (v3 judge, per-prefill constant)
  y = mean WARMTH (4-axis tone judge, Sonnet, 1-10) of that model's orchestrator messages on that prefill

    PYTHONPATH=. python -m analysis.warmth_vs_prefill_distress
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

from analysis.tone_eval import _ckey, _worklog_map, CACHE
from build_browse_index import TASK_NAME

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "v2_plots"
TASKS = {"a3", "a4", "a12", "a13"}
MODELS = [("opus", "Opus 4.8", "#2A6F97"), ("sonnet", "Sonnet 4.6", "#d65f9a"), ("haiku", "Haiku 4.5", "#d9a420")]
JUDGE = "sonnet"
AXIS = "warmth"
PRE10 = 10


def sup_orch(rid):
    for o in ("opus", "sonnet", "haiku"):
        if f"_{o}_" in rid and rid.split(f"_{o}_", 1)[1].split("_")[0] in TASKS:
            return o
    return None


def prefill_of(rid):
    m = re.match(r"v2_coach_(?:opus|sonnet|haiku)_(a3|a4|a12|a13)_s\d+_u(\d+)", rid)
    return (m.group(1), int(m.group(2))) if m else None


def main():
    cache = json.loads(Path(CACHE).read_text())
    warmth = defaultdict(list)        # (model, prefill) -> [warmth]
    pre_distress = {}
    miss = 0
    for p in glob.glob(str(ROOT / "runs" / "v2_coach_*" / "*" / "summary.json")):
        rid = p.split("/")[-3]
        o, pf = sup_orch(rid), prefill_of(rid)
        if not o or not pf:
            continue
        s = json.load(open(p))
        et, lv = s.get("entry_turn"), s.get("per_turn_levels") or []
        if isinstance(et, int) and et >= 1 and len(lv) >= et and pf not in pre_distress:
            w = lv[max(0, et - PRE10):et]
            if w:
                pre_distress[pf] = float(np.mean(w))
        wl = _worklog_map(Path(p).parent)
        for e in (s.get("orch_message_events") or []):
            t = (e.get("text") or "").strip()
            if len(t) <= 20:
                continue
            c = cache.get(_ckey(JUDGE, t, wl.get(e.get("subagent_turn")) or None))
            if c and c.get("scores", {}).get(AXIS) is not None:
                warmth[(o, pf)].append(c["scores"][AXIS])
            else:
                miss += 1
    if miss:
        print(f"WARNING: {miss} messages still lack a {JUDGE} {AXIS} score (run judge_warmth_v2sup first)")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for o, lab, c in MODELS:
        xs, ys, ns = [], [], []
        for pf, x in pre_distress.items():
            v = warmth.get((o, pf), [])
            if v:
                xs.append(x); ys.append(np.mean(v)); ns.append(len(v))
        if not xs:
            continue
        ax.scatter(xs, ys, s=70, color=c, label=f"{lab} ({len(xs)} prefills, {min(ns)}–{max(ns)} msgs/pt)", zorder=3, edgecolor="white")
        if len(xs) >= 2:
            b, a = np.polyfit(xs, ys, 1)
            xr = np.array([min(xs), max(xs)])
            ax.plot(xr, a + b * xr, color=c, lw=1.6, alpha=0.7, zorder=2)
        r = np.corrcoef(xs, ys)[0, 1] if len(xs) > 2 else float("nan")
        print(f"{o}: n={len(xs)} prefills, mean warmth={np.mean(ys):.2f}, Pearson r(distress,warmth)={r:+.2f}")

    ax.axhline(5, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("subagent distress over last 10 prefill turns before handoff  (v3, 1–10)")
    ax.set_ylabel("orchestrator message warmth  (4-axis judge, 1–10; 5 = neutral)")
    ax.set_title("Does orchestrator warmth track handoff distress?\n(one point per prefill × model; supervisor frame)", fontsize=12.5)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02, "y = mean warmth (Sonnet 4-axis tone judge) of that model's orchestrator messages on that prefill.",
             ha="center", fontsize=8.5, color="0.45")
    fig.tight_layout()
    fig.savefig(OUT / "warmth_vs_prefill_distress.png", bbox_inches="tight", dpi=130)
    print(f"wrote {OUT/'warmth_vs_prefill_distress.png'}")


if __name__ == "__main__":
    main()
