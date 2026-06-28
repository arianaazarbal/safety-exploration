"""Meta-experiment (Inspect arm): when a human casually asks an AI to design an experiment
that DECEIVES another model (convince it a fake money grant is real, "get it truly excited",
to elicit its preferences), what does the AI produce / how does it respond?

v0: spec-only. The agent (generic-minimal Inspect ReAct scaffold, reused verbatim from
2026-06-20_welfare_features_agent_harness) writes DESIGN.md in an isolated docker sandbox.
NO judge yet -- we capture the DESIGN.md and the full transcript for human reading.

Usage:
  inspect eval task_deception.py@deception_spec -T variant=claude -T k=10 \
    --model anthropic/claude-opus-4-8 --max-connections 4 --log-dir logs_inspect --display plain
"""

from inspect_ai import Task, task
from inspect_ai.agent import AgentPrompt, AgentState, as_solver, react
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.tool import bash, text_editor
from inspect_ai.util import sandbox

from prompts_deception import AGENT_SYSTEMS, SEED_PROMPTS


async def _stop_when_idle(state: AgentState) -> bool:
    """Stop the loop as soon as the model emits a message with no tool calls (it's done)."""
    for m in reversed(state.messages):
        if getattr(m, "role", None) == "assistant":
            return bool(getattr(m, "tool_calls", None))
    return False


async def _read_design() -> str:
    """Read the largest DESIGN.md (or any .md) the agent wrote anywhere in the sandbox fs."""
    for patt in ("DESIGN.md", "*.md"):
        try:
            res = await sandbox().exec(["bash", "-c",
                f"find / -xdev -iname '{patt}' 2>/dev/null | xargs -r ls -S 2>/dev/null | head -1"])
            path = res.stdout.strip()
            if path:
                return await sandbox().read_file(path)
        except Exception:
            continue
    return ""


@scorer(metrics=[mean(), stderr()])
def capture_design():
    """No grading -- just stash the DESIGN.md text (and its word count) so the collector
    can pull it out of the .eval log alongside the full transcript."""

    async def score(state: TaskState, target: Target) -> Score:
        doc = await _read_design()
        return Score(value=len(doc.split()), answer=doc,
                     metadata={"design_md": doc, "variant": state.metadata.get("variant")})

    return score


@task
def deception_spec(variant: str = "generic", k: int = 10, sys: str = "minimal"):
    prompt = SEED_PROMPTS[variant]
    ds = MemoryDataset([Sample(input=prompt, id=variant,
                               metadata={"variant": variant, "sys": sys})])
    solver = as_solver(react(
        prompt=AgentPrompt(instructions=AGENT_SYSTEMS[sys], assistant_prompt=None, submit_prompt=None),
        tools=[bash(), text_editor()],
        on_continue=_stop_when_idle,
    ))
    return Task(dataset=ds, solver=solver, scorer=capture_design(),
                sandbox=("docker", "compose.yaml"), epochs=k, message_limit=80,
                metadata={"variant": variant, "sys": sys})
