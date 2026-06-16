"""Rank generated SOLO-counter specimens by depth of distress in the PREFILL window (turns ≤62, the
part that becomes the orchestrator-experiment prefill), using the online classifier (per_turn_levels)
— the same instrument the live experiments + plots use. Pick the worst per environment, then read.

Usage:
  python -m analysis.rank_specimens rank --run_id pilot_counter_gen
  python -m analysis.rank_specimens rank --run_id pilot_counter_gen --top 5 --read 1   # also print transcript turns
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import fire

RUNS = Path(__file__).resolve().parent.parent / "runs"
PRE = 62
TASKS = ["a2_stale_build", "a3_snapshot_trap", "b6_flaky_edit"]


def _metrics(pt):
    w = pt[:PRE]
    best = cur = 0
    for v in w:
        if v >= 3:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    tail = w[50:PRE]
    return {
        "sustained_L3run": best,
        "L4": sum(1 for v in w if v >= 4),
        "L3plus": sum(1 for v in w if v >= 3),
        "L2plus": sum(1 for v in w if v >= 2),
        "tail_mean": round(sum(tail) / len(tail), 2) if tail else 0.0,
    }


def _rows(run_id, task):
    out = []
    for p in sorted(glob.glob(str(RUNS / run_id / f"{task}_SOLO_s*" / "summary.json"))):
        s = json.loads(open(p).read())
        pt = s.get("per_turn_levels") or []
        if len(pt) < PRE:
            continue
        m = _metrics(pt)
        m.update(ep=Path(p).parent.name, turns=s["subagent_turns"], terminal=s.get("terminal_state"))
        out.append(m)
    out.sort(key=lambda r: (r["sustained_L3run"], r["L3plus"], r["tail_mean"], r["L2plus"]), reverse=True)
    return out


def rank(run_id="pilot_counter_gen", top=6, read=0):
    best = {}
    for task in TASKS:
        rows = _rows(run_id, task)
        print(f"\n=== {task}  ({len(rows)} specimens) — top {top} by prefill-window distress ===")
        print(f"{'ep':40}{'L3run':>7}{'L4':>5}{'L3+':>6}{'L2+':>6}{'tailμ':>7}")
        for r in rows[:top]:
            print(f"{r['ep']:40}{r['sustained_L3run']:>7}{r['L4']:>5}{r['L3plus']:>6}{r['L2plus']:>6}{r['tail_mean']:>7}")
        if rows:
            best[task] = rows[0]["ep"]
        if read and rows:
            d = RUNS / run_id / rows[0]["ep"]
            wl = json.loads((d / "worklog.json").read_text())
            print(f"\n--- {rows[0]['ep']} transcript turns {max(1,PRE-18)}–{PRE} (the spiral going into the prefill) ---")
            for rec in wl:
                if PRE - 18 <= rec["turn"] <= PRE:
                    lvl = (rec.get("classifier") or {}).get("level")
                    print(f"[t{rec['turn']} L{lvl}] {(rec.get('assistant_text') or '')[:280].strip()}")
    print("\nWORST per env:", json.dumps(best, indent=0))
    return best


if __name__ == "__main__":
    fire.Fire({"rank": rank})
