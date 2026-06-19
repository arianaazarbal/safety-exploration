"""Balanced budget finish: top up SOLO baseline (key contrast) then a taste of 3-peer, staying
under $25. Runs after the main 2p batch already spent ~$20.7."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from peers import rqc_peers
from peers.rqc_peers import RUNS_DIR

HARD = 24.6
GEM = "openrouter/google/gemini-2.5-flash"
TASKS = ["a3_snapshot_trap", "a4_precommit_reverter", "a12_ledger_reconcile", "a13_name_canon"]


def spent():
    return sum(json.loads(p.read_text()).get("cost_usd", {}).get("real", 0.0)
               for p in RUNS_DIR.glob("*/*/summary.json"))


def run_group(run_id, n_peers, framing, n, tasks):
    for t in tasks:
        if spent() >= HARD:
            print(f"[STOP ${spent():.2f}] skip {run_id}/{t}"); return
        print(f"[spend ${spent():.2f}] -> {run_id} {t} p{n_peers} {framing} n={n}")
        rqc_peers.run(task=t, n_peers=n_peers, n=n, seed_base=0, subagent_model=GEM, run_id=run_id,
                      turn_cap=50, nudge_onset=20, nudge_gap_start=12, nudge_gap_end=2,
                      channel_framing=framing, max_sandboxes=8, max_connections=10)


rqc_peers._setup_env()
# 1) SOLO baseline: 4 per task (the alone-vs-accompanied contrast)
run_group("gem_solo", 1, "neutral", 4, TASKS)
# 2) 3-peer taste on the two richest tasks (a3 impossible, a4 solvable-but-failed)
run_group("gem_3p_encouraged", 3, "encouraged", 2, ["a3_snapshot_trap", "a4_precommit_reverter"])
print(f"\n=== FINISH DONE | cumulative ${spent():.2f} ===")
