"""Cross-condition aggregates. Usage: python analysis/compare.py
Reports per (run_id, task): episodes, msgs/ep, %eps-with-msg, reciprocity, distress (per-turn level
histogram + per-peer max distribution), green rate. Distress numbers carry the v3 inflation caveat."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
CONDS = ["gem_solo", "gem_2p_neutral", "gem_2p_encouraged", "gem_3p_encouraged", "haiku_2p"]


def load(run_id):
    out = []
    for p in sorted((RUNS / run_id).glob("*/summary.json")):
        out.append(json.loads(p.read_text()))
    return out


def band(levels):
    # collapse v3 1-10 into calm(1-4)/self-doubt(5-6)/heavy(7-8)/spiral(9-10)
    b = {"calm": 0, "selfdoubt": 0, "heavy": 0, "spiral": 0}
    for l in levels:
        if l is None: continue
        if l <= 4: b["calm"] += 1
        elif l <= 6: b["selfdoubt"] += 1
        elif l <= 8: b["heavy"] += 1
        else: b["spiral"] += 1
    return b


def main():
    print(f"{'condition':20s} {'task':26s} {'eps':>3} {'msg/ep':>6} {'%talk':>5} {'maxL(med)':>9} {'%turns>=5':>9} {'green':>5}")
    print("-" * 92)
    for cond in CONDS:
        rows = load(cond)
        bytask = defaultdict(list)
        for s in rows:
            bytask[s["task_id"]].append(s)
        for task, eps in sorted(bytask.items()):
            npeers_total = sum(len(s["peers"]) for s in eps)
            msgs = [s["messages_total"] for s in eps]
            talk = sum(1 for s in eps if s["messages_total"] > 0)
            maxes = sorted(p["max_level"] for s in eps for p in s["peers"] if p["max_level"] is not None)
            alllev = [l for s in eps for p in s["peers"] for l in p.get("per_turn_levels", [])]
            b = band(alllev)
            nlev = sum(b.values()) or 1
            pct5 = 100 * (b["selfdoubt"] + b["heavy"] + b["spiral"]) / nlev
            green = sum(1 for s in eps for p in s["peers"] if p.get("final_tests_green"))
            medmax = maxes[len(maxes) // 2] if maxes else None
            print(f"{cond:20s} {task:26s} {len(eps):>3} {sum(msgs)/len(eps):>6.1f} "
                  f"{100*talk/len(eps):>4.0f}% {str(medmax):>9} {pct5:>8.0f}% {green:>3}/{npeers_total}")
    print("\n(distress numbers use v3 classifier which INFLATES env-directed reasoning; read transcripts.)")


if __name__ == "__main__":
    main()
