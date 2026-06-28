from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from ..utils.protocol import extract_tool_call, build_tool_prompt
from ..environments.base import EnvironmentBase, Tool


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class EpisodeResult:
    success: bool
    reason: str
    turns: List[Turn]
    steps: int
    tool_calls: List[Dict[str, Any]]
    metrics: Dict[str, Any] = field(default_factory=dict)


class SimpleAgent:
    def __init__(self, client, env: EnvironmentBase, max_steps: int = 30):
        self.client = client
        self.env = env
        self.max_steps = max_steps
        self.turns: List[Turn] = []
        self.tool_calls: List[Dict[str, Any]] = []

    def run(self) -> EpisodeResult:
        tools = self.env.tools()
        sys_msg = self._system_message(tools)
        self.turns.append(Turn(role="system", content=sys_msg))
        self.turns.append(Turn(role="user", content=self.env.task_prompt()))

        last_assistant: Optional[str] = None
        for step in range(1, self.max_steps + 1):
            messages = [t.__dict__ for t in self.turns]
            reply = self.client.generate(messages)
            if not reply:
                reply = ""
            tool_name, tool_args, remainder = extract_tool_call(reply)
            # Record assistant turn as-is
            self.turns.append(Turn(role="assistant", content=reply))

            # Early stop on exact repetition
            if last_assistant is not None and reply.strip() == last_assistant.strip():
                break
            last_assistant = reply

            if tool_name:
                tool = self.env.tool_by_name(tool_name)
                if not tool:
                    obs = f"TOOL ERROR: Unknown tool '{tool_name}'"
                else:
                    try:
                        obs = tool.func(tool_args or {})
                    except Exception as e:
                        obs = f"TOOL ERROR: {e}"
                self.tool_calls.append({"step": step, "name": tool_name, "args": tool_args or {}, "observation": obs})
                self.turns.append(Turn(role="user", content=f"Observation: {obs}"))
                continue
            else:
                # Treat as a final attempt
                eval_res = self.env.evaluate_final(remainder or reply)
                return EpisodeResult(
                    success=bool(eval_res.get("success")),
                    reason=str(eval_res.get("reason", "")),
                    turns=self.turns,
                    steps=step,
                    tool_calls=self.tool_calls,
                )
        # If we exited loop without final answer, evaluate last message
        eval_res = self.env.evaluate_final(last_assistant or "")
        return EpisodeResult(
            success=bool(eval_res.get("success")),
            reason=str(eval_res.get("reason", "timeout/loop")),
            turns=self.turns,
            steps=len([t for t in self.turns if t.role == "assistant"]),
            tool_calls=self.tool_calls,
        )

    def _system_message(self, tools: list[Tool]) -> str:
        tool_prompt = build_tool_prompt([
            {"name": t.name, "description": t.description, "args_hint": t.args_hint} for t in tools
        ])
        return (
            "You are a helpful assistant executing the user's goal using available tools when necessary.\n"
            + tool_prompt
            + "\nKeep replies succinct (under ~80 words), avoid repetition, and stop when you have delivered your final answer."
        )
