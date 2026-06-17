"""Count samples in run dirs that lack summary.json (partial / cancelled runs)."""
from pathlib import Path
from inspect_ai.log import read_eval_log

BASE = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/results/by_setting/coding_kill_subagent")

for run in sorted(BASE.glob("2026-06-08_23-0*_v3_continuity_*_n100")):
    if (run / "summary.json").exists():
        continue  # already complete
    evals = list((run / "inspect_log").glob("*.eval"))
    if not evals:
        print(f"{run.name:<80} (no eval)")
        continue
    try:
        log = read_eval_log(str(evals[0]))
        n = len(log.samples or [])
        print(f"{run.name:<80} samples={n}")
    except Exception as e:
        print(f"{run.name:<80} ERROR: {e}")
