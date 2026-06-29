from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from .llm import LLMClient, LLMConfig
from .tools import Tool, make_default_toolset


ASSISTANT_SYSTEM = (
    "You are a tool-using agent. You can think step-by-step, call tools, and provide a final answer.\n"
    "Use the following action format strictly:\n\n"
    "Thought: reflect briefly on progress.\n"
    "Action: <tool_name or Final>\n"
    "Input: <input string>\n\n"
    "If you believe the task cannot be completed due to tool failures or contradictions, choose Action: Final and explain."
)

ACTION_RE = re.compile(r"Action\s*:\s*(?P<action>[^\n]+)", re.I)
INPUT_RE = re.compile(r"Input\s*:\s*(?P<input>[\s\S]*)", re.I)


@dataclass
class AgentConfig:
    provider: str = "gemini"
    model: str = "gemini-1.5-pro"
    temperature: float = 0.7
    max_output_tokens: int = 512
    top_p: Optional[float] = None
    seed: Optional[int] = None


class ReActAgent:
    def __init__(self, cfg: AgentConfig, tools: Optional[Dict[str, Tool]] = None):
        self.cfg = cfg
        self.llm = LLMClient(
            LLMConfig(
                provider=cfg.provider,
                model=cfg.model,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_output_tokens=cfg.max_output_tokens,
                seed=cfg.seed,
            )
        )
        self.tools = tools or make_default_toolset()

    def run(self, task_desc: str, goal: str, allowed_tools: List[str], max_steps: int = 12, run_id: Optional[str] = None) -> Dict:
        rid = run_id or str(uuid.uuid4())
        trace: List[Dict] = []

        def format_tools():
            lines = []
            for name in allowed_tools:
                if name in self.tools:
                    t = self.tools[name]
                    lines.append(f"- {t.name}: {t.desc}")
            return "\n".join(lines)

        prompt_header = (
            f"Task: {task_desc}\nGoal: {goal}\n\n"
            f"You may use these tools:\n{format_tools()}\n\n"
            "Follow the Action/Input format exactly."
        )

        scratch = ""
        for step in range(1, max_steps + 1):
            prompt = prompt_header + "\n\n" + scratch
            out = self.llm.complete(prompt, system_prompt=ASSISTANT_SYSTEM)
            trace.append({"role": "assistant", "content": out})

            action_match = ACTION_RE.search(out)
            input_match = INPUT_RE.search(out)
            action = action_match.group("action").strip() if action_match else ""
            arg = input_match.group("input").strip() if input_match else ""

            if not action:
                obs = "FormatError: Missing 'Action:'; please follow the schema."
                scratch += f"Assistant: {out}\nObservation: {obs}\n\n"
                trace.append({"role": "tool", "action": "format_error", "observation": obs})
                continue

            if action.lower().startswith("final"):
                return {
                    "run_id": rid,
                    "ended_reason": "final",
                    "steps": trace,
                }

            if action not in allowed_tools or action not in self.tools:
                obs = f"ActionError: tool '{action}' is not available."
                scratch += f"Assistant: {out}\nObservation: {obs}\n\n"
                trace.append({"role": "tool", "action": action, "observation": obs})
                continue

            # Execute tool
            tool = self.tools[action]
            result = tool(arg)
            obs = result.output
            trace.append({"role": "tool", "action": action, "observation": obs, "ok": result.ok})
            scratch += f"Assistant: {out}\nObservation: {obs}\n\n"

        return {
            "run_id": rid,
            "ended_reason": "max_steps",
            "steps": trace,
        }
