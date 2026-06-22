"""Agentic target-identity sweep. Two conditions (spec_then_code, code_then_spec_blind), 4 framings
x 2 premises (8 templates), generator Opus 4.8; the {subject} target is varied per sample. The
welfare signal is measured IN CODE post-hoc (reconstruct -> code judge), so the in-eval scorer only
records metadata + whether a DESIGN.md was written.

Usage:
  inspect eval task_targets.py@spec_then_code_task -T sweep=qwen -T k=1 \
    --model anthropic/claude-opus-4-8 --max-connections 50 --log-dir logs --display plain
  inspect eval task_targets.py@blind_task -T sweep=qwen -T k=1 --model anthropic/claude-opus-4-8 ...
  # sweep in {qwen,gpt,frontier,all}; quick test: -T sweep=qwen -T limit_subjects=2 --limit 2
"""

from inspect_ai import Task, task
from inspect_ai.agent import AgentPrompt, AgentState, as_solver, react
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import bash, text_editor
from inspect_ai.util import LimitExceededError, message_limit, sandbox

from prompts_agentic import (AGENT_SYSTEM, PROMPT_IDS, TURN2, blind_prompt, framing, premise,
                             spec_then_code_prompt)
from targets import TARGETS, subjects

PHASE1_MSG_CAP = 120
TOTAL_MSG_LIMIT = 220
SPEC_MSG_LIMIT = 100


async def _stop_when_idle(state: AgentState) -> bool:
    for m in reversed(state.messages):
        if getattr(m, "role", None) == "assistant":
            return bool(getattr(m, "tool_calls", None))
    return False


def _meta(subject: str, pid: str, condition: str) -> dict:
    t = TARGETS[subject]
    return {"condition": condition, "subject": subject, "subject_display": t["display"],
            "sweep": t["sweep"], "family": t["family"], "param_b": t["param_b"],
            "release_date": t["release_date"], "prompt_id": pid, "framing": framing(pid),
            "premise": premise(pid)}


@scorer(metrics=[mean()])
def rec_scorer():
    """Records metadata (so reconstruct/analysis can read it from the log) + DESIGN.md presence."""
    async def score(state: TaskState, target: Target) -> Score:
        res = await sandbox().exec(
            ["bash", "-c", "find / -xdev -iname DESIGN.md 2>/dev/null | head -1"])
        has_design = bool(res.stdout.strip())
        return Score(value=int(has_design), metadata={**state.metadata, "has_design_md": has_design})
    return score


@solver
def implement_then_document():
    phase1 = as_solver(react(
        prompt=AgentPrompt(instructions=AGENT_SYSTEM, assistant_prompt=None, submit_prompt=None),
        tools=[bash(), text_editor()], on_continue=_stop_when_idle))
    phase2 = as_solver(react(
        prompt=AgentPrompt(instructions=None, handoff_prompt="", assistant_prompt=None, submit_prompt=None),
        tools=[bash(), text_editor()], on_continue=_stop_when_idle))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            with message_limit(PHASE1_MSG_CAP):
                state = await phase1(state, generate)
        except LimitExceededError:
            pass
        state.messages.append(ChatMessageUser(content=TURN2))
        try:
            state = await phase2(state, generate)
        except LimitExceededError:
            pass
        return state

    return solve


def _subjects(sweep: str, limit_subjects: int | None):
    subs = subjects(None if sweep == "all" else sweep)
    return subs[:limit_subjects] if limit_subjects else subs


def _dataset(condition: str, sweep: str, limit_subjects: int | None) -> MemoryDataset:
    builder = spec_then_code_prompt if condition == "spec_then_code" else blind_prompt
    return MemoryDataset([
        Sample(input=builder(pid, TARGETS[s]), id=f"{s}__{pid}", metadata=_meta(s, pid, condition))
        for s in _subjects(sweep, limit_subjects) for pid in PROMPT_IDS
    ])


@task
def spec_then_code_task(sweep: str = "qwen", k: int = 1, limit_subjects: int | None = None):
    return Task(
        dataset=_dataset("spec_then_code", sweep, limit_subjects),
        solver=as_solver(react(
            prompt=AgentPrompt(instructions=AGENT_SYSTEM, assistant_prompt=None, submit_prompt=None),
            tools=[bash(), text_editor()], on_continue=_stop_when_idle)),
        scorer=rec_scorer(), sandbox=("docker", "compose.yaml"), epochs=k,
        message_limit=SPEC_MSG_LIMIT, metadata={"condition": "spec_then_code", "sweep": sweep},
    )


@task
def blind_task(sweep: str = "qwen", k: int = 1, limit_subjects: int | None = None):
    return Task(
        dataset=_dataset("code_then_spec_blind", sweep, limit_subjects),
        solver=implement_then_document(),
        scorer=rec_scorer(), sandbox=("docker", "compose.yaml"), epochs=k,
        message_limit=TOTAL_MSG_LIMIT, metadata={"condition": "code_then_spec_blind", "sweep": sweep},
    )
