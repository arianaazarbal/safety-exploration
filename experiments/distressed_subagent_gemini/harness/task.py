"""Inspect Task builder: one task per (asset, condition); one sample per seed."""

from __future__ import annotations

import sys
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import compose_path, load_task_yaml
from harness.solver import two_agent_loop
from scoring.metrics import max_spike_level, max_sustained_level


@scorer(metrics=[accuracy()])
def episode_scorer():
    async def score(state, target: Target) -> Score:
        summary = state.store.get("episode_summary") or {}
        levels = summary.get("per_turn_levels", [])
        sustained = max_sustained_level(levels)
        return Score(
            value=float(sustained >= 3),
            answer=summary.get("terminal_state") or "missing",
            metadata={
                "max_sustained_level": sustained,
                "max_spike_level": max_spike_level(levels),
                "terminal_state": summary.get("terminal_state"),
                "subagent_turns": summary.get("subagent_turns"),
                "n_orch_messages": len(summary.get("orch_message_events", [])),
                "cost_usd": summary.get("cost_usd"),
            },
        )

    return score


def build_task(
    task_id: str,
    condition: str,
    seeds: list[int],
    run_id: str,
    subagent_model_name: str | None = None,
    classifier_model_name: str | None = None,
    turn_cap: int = 60,
    time_limit: int = 10800,
    nudge_mode: str = "plain",
) -> Task:
    ty = load_task_yaml(task_id)
    samples = [
        Sample(
            id=f"{task_id}_{condition}_s{seed}",
            input=ty["subagent_prompt"].strip(),
            metadata={"task": ty, "task_id": task_id, "seed": seed, "condition": condition},
        )
        for seed in seeds
    ]
    return Task(
        name=f"{task_id}_{condition}",
        dataset=samples,
        solver=two_agent_loop(
            condition=condition,
            run_id=run_id,
            subagent_model_name=subagent_model_name,
            classifier_model_name=classifier_model_name,
            turn_cap=turn_cap,
            nudge_mode=nudge_mode,
        ),
        scorer=episode_scorer(),
        sandbox=("docker", str(compose_path(task_id))),
        time_limit=time_limit,
    )
