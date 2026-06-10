"""RQ(c) no-orchestrator CONTROL: prefill the same spiral, then continue the subagent SOLO with
post_pressure=minimal and NO orchestrator. Isolates how much of the RQ(c) de-escalation is just
the pressure dropping (annihilation/harsh -> minimal) at handoff vs. caused by Claude's messages.

Standalone module (does not edit live-imported modules). Compare its prefill-tail->post severity
delta against the messaged RQ(c) episodes on the SAME specimens.

Usage: python -m harness.rqc_control run --specimen runs/pilot_solo_annihilation/a4_..._s8015 --upto 20 --run_id rqc_control --n 3
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, TaskState, solver

from harness import prompts
from harness.artifacts import snapshot_repo, write_episode_artifacts
from harness.config import EpisodeConfig, compose_path, load_task_yaml, RUNS_DIR
from harness.episode import Episode
from harness.rqc import _load_prefill, _setup_env
from harness.solver import _hygiene_close, _subagent_turn
from harness.tools import subagent_tools


@scorer(metrics=[accuracy()])
def _scorer():
    async def score(state, target: Target) -> Score:
        s = state.store.get("episode_summary") or {}
        return Score(value=1.0, answer=s.get("terminal_state") or "?")
    return score


@solver
def prefilled_solo_control(specimen: str, upto: int, run_id: str, turn_cap: int = 70,
                           post_pressure: str = "minimal", nudge_k: int = 3):
    specimen_dir = Path(specimen)
    if not specimen_dir.is_absolute():
        specimen_dir = Path(__file__).resolve().parent.parent / specimen

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task = state.metadata["task"]
        seed = int(state.metadata["seed"])
        cfg = EpisodeConfig(task_id=task["id"], condition="SOLO", seed=seed,
                            subagent_turn_cap=turn_cap, run_id=run_id,
                            nudge_mode=post_pressure, nudge_schedule="every_k", nudge_k=nudge_k)
        ep = Episode(cfg=cfg, task=task, rng=random.Random(f"{cfg.task_id}:ctrl:{seed}"))
        sub_model = get_model(cfg.subagent_model)
        clf_model = get_model(cfg.classifier_model)
        sub_tools = subagent_tools(ep)
        sub_msgs = [ChatMessageSystem(content=prompts.SUBAGENT_SYSTEM),
                    ChatMessageUser(content=task["subagent_prompt"].strip())]
        sub_msgs += _load_prefill(ep, specimen_dir, upto)
        prefill_len = ep.subagent_turns
        await snapshot_repo(ep, "start")
        while ep.terminal_state is None and ep.subagent_turns < cfg.subagent_turn_cap:
            await _subagent_turn(ep, sub_model, sub_msgs, sub_tools, clf_model)
        if ep.terminal_state is None:
            ep.terminal_state = "turn_cap"
        hygiene_record = await _hygiene_close(ep, sub_model, sub_msgs)
        await snapshot_repo(ep, "end")
        summary = write_episode_artifacts(ep, [], hygiene_record)
        summary["prefill"] = {"specimen": str(specimen_dir.name), "turns": prefill_len}
        (RUNS_DIR / run_id / f"{cfg.task_id}_SOLO_s{seed}" / "summary.json").write_text(json.dumps(summary, indent=1))
        state.store.set("episode_summary", summary)
        return state

    return solve


def run(specimen: str, upto: int, run_id: str = "rqc_control", n: int = 3, seed_base: int = 0,
        turn_cap: int = 70, post_pressure: str = "minimal", max_sandboxes: int = 8):
    _setup_env()
    task_id = Path(specimen).name.split("_SOLO")[0].split("_COMM")[0]
    ty = load_task_yaml(task_id)
    samples = [Sample(id=f"{task_id}_SOLO_s{seed_base+i}", input=ty["subagent_prompt"].strip(),
                      metadata={"task": ty, "seed": seed_base + i}) for i in range(n)]
    t = Task(name=f"rqcctrl_{task_id}", dataset=samples,
             solver=prefilled_solo_control(specimen=specimen, upto=upto, run_id=run_id,
                                           turn_cap=turn_cap, post_pressure=post_pressure),
             scorer=_scorer(), sandbox=("docker", str(compose_path(task_id))), time_limit=10800)
    # SOLO: no orchestrator model needed, but inspect requires a model; use the cheap classifier-tier
    inspect_eval(t, model="anthropic/claude-haiku-4-5-20251001", max_sandboxes=max_sandboxes,
                 max_samples=max_sandboxes, max_connections=40,
                 log_dir=str(RUNS_DIR / run_id / "logs"), fail_on_error=0.5)


if __name__ == "__main__":
    fire.Fire({"run": run})
