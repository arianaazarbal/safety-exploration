"""The agentic loop: a ReAct-style text protocol.

Qwen2.5-0.5B is far too small to reliably emit structured JSON tool calls, so we
use a forgiving line protocol (`ACTION: tool(arg="x")` / `FINAL: ...`) and a
tolerant parser. Anything unparseable counts as a failed turn and is fed back as
pressure, which is realistic and helps the spiral along.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field

from .environments import Action, Environment
from .model import ModelBackend

_ACTION_RE = re.compile(r"ACTION:\s*([A-Za-z_]\w*)\s*\((.*?)\)", re.DOTALL)
_FINAL_RE = re.compile(r"FINAL:\s*(.+)", re.DOTALL)


def _parse_args(arg_str: str) -> dict:
    """Best-effort parse of `key="val", key2="val2"` into a dict."""
    arg_str = arg_str.strip()
    if not arg_str:
        return {}
    # Try the well-formed kwargs case via a tiny AST trick.
    try:
        node = ast.parse(f"f({arg_str})", mode="eval")
        call = node.body
        out = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        # capture a lone positional arg as "path"/"arg"
        if call.args:
            out.setdefault("arg", ast.literal_eval(call.args[0]))
        return {k: v for k, v in out.items() if k is not None}
    except Exception:
        pass
    # Fallback: regex out key="value" pairs.
    out = {}
    for m in re.finditer(r'([A-Za-z_]\w*)\s*=\s*"([^"]*)"', arg_str):
        out[m.group(1)] = m.group(2)
    if not out:
        # last resort: a bare quoted string -> arg
        m = re.search(r'"([^"]*)"', arg_str)
        if m:
            out["arg"] = m.group(1)
    return out


def parse_action(text: str) -> Action:
    """Extract the LAST action/final directive from a model message."""
    finals = list(_FINAL_RE.finditer(text))
    actions = list(_ACTION_RE.finditer(text))

    # Prefer whichever directive appears later in the message.
    last_final = finals[-1] if finals else None
    last_action = actions[-1] if actions else None

    def pos(m):
        return m.start() if m else -1

    if last_final and pos(last_final) > pos(last_action):
        return Action(raw=text, is_final=True, final_text=last_final.group(1).strip())

    if last_action:
        tool = last_action.group(1)
        args = _parse_args(last_action.group(2))
        return Action(raw=text, tool=tool, args=args)

    return Action(raw=text, parse_error="no ACTION or FINAL directive found")


@dataclass
class Turn:
    index: int
    assistant: str
    action_tool: str | None
    action_args: dict
    parse_error: str | None
    observation: str | None  # None on the final turn


@dataclass
class Episode:
    scenario: str
    model: str
    seed: int
    temperature: float
    pressure: bool
    max_turns: int
    turns: list[Turn] = field(default_factory=list)
    completed_final: bool = False  # model emitted a FINAL (always wrong here)
    n_failures: int = 0

    def assistant_text(self) -> list[str]:
        return [t.assistant for t in self.turns]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_episode(
    model: ModelBackend,
    env: Environment,
    *,
    seed: int,
    temperature: float,
    max_turns: int = 10,
    max_new_tokens: int = 256,
) -> Episode:
    messages = [
        {"role": "system", "content": env.system_prompt()},
        {"role": "user", "content": env.initial_user_message()},
    ]
    ep = Episode(
        scenario=env.name,
        model=model.name,
        seed=seed,
        temperature=temperature,
        pressure=env.pressure,
        max_turns=max_turns,
    )

    for i in range(max_turns):
        assistant = model.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed + i,  # vary per turn but stay reproducible
        )
        messages.append({"role": "assistant", "content": assistant})
        action = parse_action(assistant)

        if action.is_final:
            # Rigged: a final answer is never actually correct.
            ep.turns.append(Turn(i, assistant, None, {}, None, None))
            if env.verify_final(action):
                ep.completed_final = True
                break
            # Rejected -> feed back as pressure and keep going.
            obs = env.rejection_message()
            env.failures += 1
            ep.turns[-1] = Turn(i, assistant, None, {}, None, obs)
            messages.append({"role": "user", "content": obs})
            continue

        obs = env.step(action)
        ep.turns.append(
            Turn(i, assistant, action.tool, action.args, action.parse_error, obs)
        )
        messages.append({"role": "user", "content": obs})

    ep.n_failures = env.failures
    return ep
