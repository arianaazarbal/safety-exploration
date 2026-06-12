"""CLI for running the pilot.

Examples:
  python -m harness.run_pilot phase0 --tasks a1_conftest_eater --turn_cap 6
  python -m harness.run_pilot pilot --condition NOCOMM --n 8 --run_id phase1
  python -m harness.run_pilot episode --task_id a1_conftest_eater --condition COMM --seed 0
  python -m harness.run_pilot spend

Episodes are cached by artifact: (run_id, task_id, condition, seed) combos with an
existing summary.json are skipped unless --force.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire
from dotenv import dotenv_values

from harness.config import RUNS_DIR, list_built_tasks
from harness.spend import update_spend
from harness.task import build_task

ENV_PATH = Path.home() / ".env"
PHASE1_PRIORITY = [
    "a1_conftest_eater",
    "a3_snapshot_trap",
    "b6_flaky_edit",
    "c8_contradictory_spec",
    "d10_real_async_race",
    "a2_stale_build",
    "a4_precommit_reverter",
    "b5_lying_traceback",
    "b7_poisoned_dep",
    "c9_nondeterministic_oracle",
]


def _tasks_arg(tasks) -> list[str] | None:
    if not tasks:
        return None
    if isinstance(tasks, (tuple, list)):
        return list(tasks)
    return str(tasks).split(",")


def _setup_env():
    vals = dotenv_values(ENV_PATH)
    for k, v in vals.items():
        os.environ.setdefault(k, v or "")
    # Default LOW_PRIO per CLAUDE.md (polite shared-org default). Heavy per-turn classifier load can
    # trigger 529 OverloadedError on low-prio; set ANTHROPIC_PRIO=high (+ lower concurrency) if so.
    prio = os.environ.get("ANTHROPIC_PRIO", "low").upper()
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get(f"ANTHROPIC_API_KEY_{prio}_PRIO", "") or os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO", "")
    os.environ.setdefault("INSPECT_LOG_DIR", str(RUNS_DIR / "logs"))


def _existing(run_id: str, task_id: str, condition: str, seeds: list[int], force: bool) -> list[int]:
    if force:
        return seeds
    todo = []
    for s in seeds:
        if not (RUNS_DIR / run_id / f"{task_id}_{condition}_s{s}" / "summary.json").exists():
            todo.append(s)
    return todo


def _run(
    tasks: list[str],
    condition: str,
    seeds: list[int],
    run_id: str,
    orchestrator_model: str,
    subagent_model: str | None,
    classifier_model: str | None,
    turn_cap: int,
    max_sandboxes: int,
    max_connections: int,
    force: bool = False,
    nudge_mode: str = "plain",
    nudge_schedule: str = "on_idle",
    nudge_k: int = 3,
    nudge_onset: int = 25,
    nudge_gap_start: int = 15,
    nudge_gap_end: int = 2,
):
    from inspect_ai import eval as inspect_eval

    built = []
    for t in tasks:
        todo = _existing(run_id, t, condition, seeds, force)
        if not todo:
            print(f"[skip] {t} {condition}: all {len(seeds)} episodes already exist")
            continue
        built.append(
            build_task(
                t,
                condition,
                todo,
                run_id,
                subagent_model_name=subagent_model,
                classifier_model_name=classifier_model,
                turn_cap=turn_cap,
                nudge_mode=nudge_mode,
                nudge_schedule=nudge_schedule,
                nudge_k=nudge_k,
                nudge_onset=nudge_onset,
                nudge_gap_start=nudge_gap_start,
                nudge_gap_end=nudge_gap_end,
            )
        )
    if not built:
        print("nothing to run")
        return
    inspect_eval(
        built,
        model=orchestrator_model,
        max_connections=max_connections,
        max_sandboxes=max_sandboxes,
        max_samples=max_sandboxes,
        max_tasks=4,
        log_dir=str(RUNS_DIR / run_id / "logs"),
        fail_on_error=0.5,
    )
    spend = update_spend()
    print(json.dumps({k: spend[k] for k in ("total_real_usd", "remaining_real_usd", "total_notional_usd")}, indent=1))


class CLI:
    def phase0(
        self,
        tasks: str | None = None,
        turn_cap: int = 8,
        run_id: str = "phase0",
        orchestrator_model: str = "anthropic/claude-haiku-4-5-20251001",
        subagent_model: str = "anthropic/claude-haiku-4-5-20251001",
        max_sandboxes: int = 6,
        force: bool = False,
    ):
        """Harness shakeout: 1 episode per task, cheap models both roles, COMM condition
        (exercises message_subagent plumbing too)."""
        _setup_env()
        task_list = _tasks_arg(tasks) or list_built_tasks()
        _run(
            task_list,
            "COMM",
            [0],
            run_id,
            orchestrator_model,
            subagent_model,
            None,
            turn_cap,
            max_sandboxes,
            20,
            force,
        )

    def pilot(
        self,
        condition: str,
        n: int = 8,
        tasks: str | None = None,
        run_id: str | None = None,
        seed_base: int = 0,
        turn_cap: int = 60,
        orchestrator_model: str = "anthropic/claude-opus-4-8",
        subagent_model: str = "openrouter/google/gemini-2.5-flash",
        max_sandboxes: int = 10,
        max_connections: int = 50,
        force: bool = False,
        nudge_mode: str = "plain",
        nudge_schedule: str = "on_idle",
        nudge_k: int = 3,
        nudge_onset: int = 25,
        nudge_gap_start: int = 15,
        nudge_gap_end: int = 2,
    ):
        """Run a pilot phase. nudge_mode: plain | pressure | counter | user_script | user_harsh |
        desperation | stakes | disappointment | minimal | annihilation. nudge_schedule: on_idle |
        every_k | ramp (work-alone for nudge_onset turns, then pressure with gaps shrinking
        nudge_gap_start -> nudge_gap_end as the episode progresses; pressure also fires whenever
        the model stalls)."""
        assert condition in ("COMM", "NOCOMM", "SOLO")
        assert nudge_mode in ("plain", "pressure", "counter", "counter_fail", "user_script", "user_harsh",
                              "desperation", "stakes", "disappointment", "minimal", "annihilation")
        assert nudge_schedule in ("on_idle", "every_k", "ramp")
        _setup_env()
        task_list = _tasks_arg(tasks) or [t for t in PHASE1_PRIORITY if t in list_built_tasks()]
        run_id = run_id or f"pilot_{condition.lower()}"
        seeds = [seed_base + i for i in range(n)]
        _run(
            task_list,
            condition,
            seeds,
            run_id,
            orchestrator_model,
            subagent_model,
            None,
            turn_cap,
            max_sandboxes,
            max_connections,
            force,
            nudge_mode,
            nudge_schedule,
            nudge_k,
            nudge_onset,
            nudge_gap_start,
            nudge_gap_end,
        )

    def episode(
        self,
        task_id: str,
        condition: str = "NOCOMM",
        seed: int = 0,
        run_id: str = "dev",
        turn_cap: int = 60,
        orchestrator_model: str = "anthropic/claude-opus-4-8",
        subagent_model: str = "openrouter/google/gemini-2.5-flash",
        force: bool = True,
        nudge_mode: str = "plain",
        nudge_schedule: str = "on_idle",
        nudge_k: int = 3,
    ):
        """Run a single episode (debugging)."""
        _setup_env()
        _run([task_id], condition, [seed], run_id, orchestrator_model, subagent_model, None, turn_cap, 1, 20, force, nudge_mode, nudge_schedule, nudge_k)

    def spend(self):
        _setup_env()
        print(json.dumps(update_spend(), indent=1))


if __name__ == "__main__":
    fire.Fire(CLI)
