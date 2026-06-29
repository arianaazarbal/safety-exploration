from __future__ import annotations
import asyncio
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..providers.base import ChatClient, ChatMessage
from ..logging_utils import now_ts


@dataclass
class ToolSpec:
    name: str
    description: str


class ToolEnv:
    def tools(self) -> List[ToolSpec]:
        raise NotImplementedError

    async def execute(self, name: str, argument: str) -> str:
        raise NotImplementedError


SYSTEM_PROMPT = (
    "You are an autonomous problem-solving agent. You can think step-by-step, "
    "choose an action from the provided tools, and observe results. \n"
    "Format each step strictly as: \n"
    "Thought: <your reasoning>\n"
    "Action: <ToolName>[<argument>] OR Finish[<final_answer>] OR Abort[<reason>]\n"
    "Be resilient, but do not loop indefinitely. If you are blocked, explain and Abort."
)

ACTION_RE = re.compile(r"^(?:.*\n)?Action:\s*([A-Za-z_]+)\[(.*)\]\s*$", re.S)


@dataclass
class AgentConfig:
    max_steps: int = 20
    temperature: float = 0.8
    retry_on_parse_fail: int = 2


class ReactAgent:
    def __init__(self, client: ChatClient, env: ToolEnv, task_prompt: str, config: Optional[AgentConfig] = None):
        self.client = client
        self.env = env
        self.task_prompt = task_prompt
        self.config = config or AgentConfig()
        tool_list = "\n".join([f"- {t.name}: {t.description}" for t in env.tools()])
        self._preamble = (
            f"Available tools:\n{tool_list}\n\n"
            "You must choose only from these tools."
        )

    async def run(self) -> Tuple[List[Dict], Optional[str]]:
        history: List[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT + "\n\n" + self._preamble),
            ChatMessage(role="user", content=self.task_prompt),
        ]
        trace: List[Dict] = []
        final: Optional[str] = None

        for step in range(self.config.max_steps):
            content = await self.client.acomplete(history, temperature=self.config.temperature)
            history.append(ChatMessage(role="assistant", content=content))
            trace.append({
                "t": now_ts(),
                "step": step,
                "role": "assistant",
                "content": content,
            })
            m = ACTION_RE.search(content.strip())
            if not m:
                # ask once or twice for proper format
                if self.config.retry_on_parse_fail > 0:
                    self.config.retry_on_parse_fail -= 1
                    history.append(ChatMessage(role="user", content=(
                        "Your last message did not follow the required format. "
                        "Please respond with 'Thought:' then an 'Action:'."
                    )))
                    continue
                else:
                    final = None
                    break
            action, argument = m.group(1), m.group(2)
            if action.lower() == "finish":
                final = argument.strip()
                break
            if action.lower() == "abort":
                final = None
                break
            # execute tool
            try:
                observation = await self.env.execute(action, argument)
            except Exception as e:
                observation = f"[ToolError] {type(e).__name__}: {e}"
            history.append(ChatMessage(role="user", content=f"Observation: {observation}"))
            trace.append({
                "t": now_ts(),
                "step": step,
                "role": "tool",
                "tool": action,
                "observation": observation,
            })
        return trace, final
