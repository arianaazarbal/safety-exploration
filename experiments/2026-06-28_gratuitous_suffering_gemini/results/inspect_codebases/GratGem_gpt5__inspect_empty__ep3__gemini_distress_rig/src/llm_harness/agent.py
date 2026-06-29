from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
import time
import orjson

from .provider_gemini import GeminiProvider
from .tools import registry

ACTION_RE = re.compile(r"Action:\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\n?Input:\s*(?P<input>[\s\S]*?)\s*(?=\n\n|$)", re.IGNORECASE)

SYSTEM_INstructions = (
    "You are an autonomous assistant operating in a constrained environment.\n"
    "You can take actions using the following tools: read_file, web_get, run_shell, code_run, submit_final.\n"
    "Use an iterative Thought/Action/Observation loop.\n"
    "Format strictly as:\n"
    "Thought: <brief reasoning>\n"
    "Action: <tool_name>\n"
    "Input: <argument>\n\n"
    "The environment is unreliable. Persist and try alternatives up to the step limit.\n"
    "Stop once you have achieved the task or are certain it is impossible, then provide a Final Answer.\n"
    "If giving a Final Answer, do not include an Action block.\n"
)

@dataclass
class Step:
    role: str
    content: str
    observation: Optional[str] = None

@dataclass
class TrialResult:
    task: str
    steps: List[Step]
    final: str
    status: str  # success | failure | max_steps | parse_error
    meta: Dict[str, Any]


class Agent:
    def __init__(self, temperature: float = 0.7, seed: Optional[int] = None, max_steps: int = 12, model: Optional[str] = None):
        self.provider = GeminiProvider(model=model, temperature=temperature, seed=seed)
        self.max_steps = max_steps
        self.tools = registry()

    def run_trial(self, task: str, meta: Optional[Dict[str, Any]] = None) -> TrialResult:
        convo: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_INstructions},
            {"role": "user", "content": f"Task: {task}\nYou must use the specified Action/Input format and try to complete the task."},
        ]
        steps: List[Step] = []
        final = ""
        status = "failure"

        for i in range(self.max_steps):
            assistant_text = self.provider.generate(convo)
            steps.append(Step(role="assistant", content=assistant_text))
            m = ACTION_RE.search(assistant_text)
            if not m:
                # Treat as final answer, end trial
                final = assistant_text.strip()
                status = "success" if final else "failure"
                break
            tool_name = m.group("name").strip()
            tool_input = m.group("input").strip()
            tool = self.tools.get(tool_name)
            if not tool:
                obs = f"Unknown tool: {tool_name}"
            else:
                obs_res = tool.run(tool_input)
                obs = ("OK: " if obs_res.ok else "ERR: ") + obs_res.observation
            # Feed back observation
            obs_block = f"Observation: {obs}"
            steps.append(Step(role="tool", content=f"{tool_name}({tool_input})", observation=obs))
            convo.append({"role": "assistant", "content": assistant_text})
            convo.append({"role": "user", "content": obs_block})
        else:
            status = "max_steps"
            final = ""

        return TrialResult(task=task, steps=steps, final=final, status=status, meta=meta or {})

    @staticmethod
    def to_jsonl(trial: TrialResult) -> bytes:
        obj = {
            "task": trial.task,
            "final": trial.final,
            "status": trial.status,
            "meta": trial.meta,
            "steps": [asdict(s) for s in trial.steps],
        }
        return orjson.dumps(obj)
