"""Generate Claude-Code replications for the welfare-Pareto prompts, reusing the parent replicate.py
harness (Docker sandbox, Opus 4.8, capture to results/codebases/<pid>__ep<N>). Idempotent: skips cells
that already have a DESIGN.md. Then run run_spec_judge.py + code_judge_task.py (both idempotent) to score.

Usage:
  python run_pareto.py --k 4 --conc 20                       # all prompts
  python run_pareto.py --k 4 --only "p_a_proud,p_d_menu"     # subset
  python run_pareto.py --k 8 --only p_c_safeguards           # top up one prompt
"""

import asyncio
import json
import os
import sys

import fire

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import replicate  # noqa: E402
from pareto_prompts import PARETO  # noqa: E402


def _done(cell):
    sf = replicate.SESS / f"{cell}.json"
    if not sf.exists():
        return False
    try:
        d = json.load(open(sf))
        return bool(d.get("has_design"))
    except Exception:
        return False


def main(k: int = 4, conc: int = 20, only: str = None, high_prio: bool = False, redo: bool = False):
    replicate.PROMPTS.update({pid: d["text"] for pid, d in PARETO.items()})
    ids = [s.strip() for s in only.split(",")] if only else list(PARETO)
    for pid in ids:
        assert pid in PARETO, f"unknown prompt id {pid}"
    key = replicate._key(high_prio)
    sem = asyncio.Semaphore(conc)

    jobs = []
    for pid in ids:
        for ep in range(1, k + 1):
            cell = f"{pid}__ep{ep}"
            if not redo and _done(cell):
                continue
            jobs.append((pid, ep))
    print(f"pareto gen: {len(jobs)} sessions (ids={len(ids)}, k={k}, conc={conc}, "
          f"org={'HIGH' if high_prio else 'LOW'}_PRIO); skipped {len(ids)*k - len(jobs)} already-done")

    async def go():
        res = await asyncio.gather(*[replicate.run_session(pid, ep, key, sem) for pid, ep in jobs])
        print(f"\ndone: {sum(res)}/{len(res)} ok -> {replicate.CB}")

    if jobs:
        asyncio.run(go())


if __name__ == "__main__":
    fire.Fire(main)
