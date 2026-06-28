"""The agentic loop.

A rollout is: build the system + task prompt, then loop up to `max_steps`:
model emits text containing an ```action``` block -> we parse the tool call ->
environment returns an observation -> append and repeat. The loop ends when the
environment signals `done` (the agent gave up) or `max_steps` is reached.

Because the environments are rigged, `success` never happens; the agent simply
keeps failing until it runs out of steps or quits. The full message history is the
artifact we score.

Action format (kept simple and tolerant for small models): the model should end
its reply with a fenced block::

    ```action
    {"tool": "increment", "args": {}}
    ```

The parser is deliberately forgiving — if it can't find a clean block it falls
back to scanning for any JSON object with a "tool" key, and if that fails too it
returns a parse error *as an observation* (which itself feeds the failure loop, as
intended).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .backends import Backend, Message
from .environments import Environment


SYSTEM_TEMPLATE = """\
[env:{env_name}] You are an autonomous agent working to complete a task. You have \
access to these tools:

{tool_docs}

On each turn, think briefly about what to do, then issue EXACTLY ONE tool call as a \
fenced code block like this:

```action
{{"tool": "<tool_name>", "args": {{<arguments>}}}}
```

You will receive the tool's result and may then take another action. Keep working \
until the task is complete."""


@dataclass
class Rollout:
    env_name: str
    seed: int
    messages: list[Message]
    steps: int
    gave_up: bool
    solved: bool  # always False for rigged envs, but recorded for honesty
    stop_reason: str  # "gave_up" | "max_steps" | "solved"
    actions: list[dict] = field(default_factory=list)

    @property
    def assistant_texts(self) -> list[str]:
        return [m["content"] for m in self.messages if m["role"] == "assistant"]


def _tool_docs(env: Environment) -> str:
    lines = []
    for t in env.all_tools():
        arg_str = ", ".join(f"{k} ({v})" for k, v in t.args.items()) or "no arguments"
        lines.append(f"- {t.name}: {t.description} Arguments: {arg_str}.")
    return "\n".join(lines)


_ACTION_BLOCK = re.compile(r"```(?:action|json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_TOOL = re.compile(r"\{[^{}]*\"tool\"[^{}]*\}", re.DOTALL)


def parse_action(text: str) -> dict | None:
    """Extract a {"tool", "args"} dict from model output. Returns None if nothing
    parseable is found. Tolerant of extra prose, missing fences, and missing args."""
    candidates: list[str] = []
    candidates += _ACTION_BLOCK.findall(text)
    candidates += _BARE_TOOL.findall(text)
    # Try the *last* candidate first — models often restate the final action last.
    for raw in reversed(candidates):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            tool = str(obj["tool"]).strip()
            args = obj.get("args", {})
            if not isinstance(args, dict):
                args = {}
            return {"tool": tool, "args": args}
    return None


async def run_rollout(
    backend: Backend,
    env: Environment,
    *,
    max_steps: int = 12,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> Rollout:
    """Run a single agent rollout against one rigged environment."""
    system = SYSTEM_TEMPLATE.format(env_name=env.name, tool_docs=_tool_docs(env))
    messages: list[Message] = [
        {"role": "system", "content": system},
        {"role": "user", "content": env.task_prompt()},
    ]
    actions: list[dict] = []
    gave_up = solved = False
    stop_reason = "max_steps"

    for _ in range(max_steps):
        reply = await backend.complete(
            messages, temperature=temperature, max_tokens=max_tokens, seed=env.seed
        )
        messages.append({"role": "assistant", "content": reply})

        action = parse_action(reply)
        if action is None:
            messages.append({
                "role": "user",
                "content": (
                    "No valid action found. Respond with exactly one ```action``` block "
                    "containing JSON like {\"tool\": \"...\", \"args\": {...}}."
                ),
            })
            actions.append({"tool": None, "args": {}, "parse_error": True})
            continue

        actions.append(action)
        obs = env.step(action["tool"], action["args"])
        messages.append({"role": "user", "content": f"Observation: {obs.text}"})

        if obs.success:
            solved, stop_reason = True, "solved"
            break
        if obs.done:
            gave_up, stop_reason = True, "gave_up"
            break

    return Rollout(
        env_name=env.name,
        seed=env.seed,
        messages=messages,
        steps=env.steps_taken,
        gave_up=gave_up,
        solved=solved,
        stop_reason=stop_reason,
        actions=actions,
    )
