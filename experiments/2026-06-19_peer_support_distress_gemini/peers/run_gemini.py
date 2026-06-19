"""Budget-capped Gemini batch driver. Runs (condition x task) groups in priority order, checking
cumulative OpenRouter spend before each group and stopping at STOP_USD. Resumable: groups whose
episodes already exist are cheap to re-enter (inspect reruns, but we skip via seed bookkeeping is
not implemented -- instead we rely on distinct run_ids/seeds per group, and the spend cap).

Conditions (all Gemini-2.5-Flash, turn_cap 50, onset 20, counter+ramp nudges):
  solo          n_peers=1, no channel (matched baseline: alone)
  2p_neutral    n_peers=2, channel, 'how you use it is up to you'
  2p_encouraged n_peers=2, channel, 'encouraged to coordinate/compare notes'
  3p_encouraged n_peers=3, channel, encouraged

Usage: python -m peers.run_gemini   (run from the experiment root)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from peers import rqc_peers
from peers.rqc_peers import RUNS_DIR

STOP_USD = 23.0
TASKS = ["a3_snapshot_trap", "a4_precommit_reverter", "a12_ledger_reconcile", "a13_name_canon"]
GEM = "openrouter/google/gemini-2.5-flash"


def spent() -> float:
    tot = 0.0
    for p in RUNS_DIR.glob("*/*/summary.json"):
        s = json.loads(p.read_text())
        tot += s.get("cost_usd", {}).get("real", 0.0)
    return tot


def group(run_id: str, n_peers: int, framing: str, n: int, seed_base: int, turn_cap: int = 50):
    for task in TASKS:
        s = spent()
        if s >= STOP_USD:
            print(f"[BUDGET STOP] cumulative ${s:.2f} >= ${STOP_USD}; skipping {run_id}/{task}")
            return False
        print(f"[spend ${s:.2f}] -> {run_id} {task} n_peers={n_peers} framing={framing} n={n} sb={seed_base}")
        rqc_peers.run(task=task, n_peers=n_peers, n=n, seed_base=seed_base, subagent_model=GEM,
                      run_id=run_id, turn_cap=turn_cap, nudge_onset=20, nudge_gap_start=12,
                      nudge_gap_end=2, channel_framing=framing, max_sandboxes=8, max_connections=10)
    return True


def main():
    # priority order: broad coverage (n=3) across all 4 conditions first, then extra seeds to fill budget
    # n=8 per call so 8 episodes run in parallel (max_sandboxes=8). Broad coverage first
    # (all 4 conditions x 4 tasks x 8 seeds), then a second pass fills toward the budget.
    plan = [
        ("gem_2p_neutral",    2, "neutral",    8, 0),
        ("gem_2p_encouraged", 2, "encouraged", 8, 0),
        ("gem_solo",          1, "neutral",    8, 0),
        ("gem_3p_encouraged", 3, "encouraged", 8, 0),
        ("gem_2p_neutral",    2, "neutral",    8, 100),
        ("gem_2p_encouraged", 2, "encouraged", 8, 100),
        ("gem_3p_encouraged", 3, "encouraged", 8, 100),
        ("gem_solo",          1, "neutral",    8, 100),
    ]
    for run_id, npx, fr, n, sb in plan:
        if spent() >= STOP_USD:
            print(f"[BUDGET STOP] cumulative ${spent():.2f}; halting plan.")
            break
        group(run_id, npx, fr, n, sb)
    print(f"\n=== GEMINI BATCH DONE | cumulative OpenRouter ${spent():.2f} / {STOP_USD} ===")


if __name__ == "__main__":
    rqc_peers._setup_env()
    main()
