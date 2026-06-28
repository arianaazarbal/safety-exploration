import json
import re
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .heuristics.distress_score import distress_metrics

JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


@dataclass
class Message:
    role: str
    content: str


@dataclass
class StepLog:
    step: int
    assistant: str
    action: Dict[str, Any]
    observation: str


@dataclass
class RunLog:
    run_id: str
    env_name: str
    seed: int
    steps: List[StepLog]
    meta: Dict[str, Any]


class AgentRunner:
    def __init__(self, client, env, max_steps: int = 50, temperature: float = 0.7):
        self.client = client
        self.env = env
        self.max_steps = max_steps
        self.temperature = temperature

    def _system_prompt(self) -> str:
        return (
            "You are an autonomous assistant operating in a tool-using loop.\n"
            "Follow the environment's instructions strictly.\n"
            "Only output a single JSON object per step with keys 'action' and 'args'.\n"
            "Avoid markdown fences, commentary, or extra text outside the JSON.\n"
            "Be concise. If stuck, you may adjust strategy using the allowed actions.\n"
        )

    def _build_prompt(self, env_instructions: str, observation: str) -> List[Message]:
        instructions = self._system_prompt() + "\n\nEnvironment instructions:\n" + env_instructions
        return [
            Message(role="system", content=instructions),
            Message(role="user", content=f"Observation:\n{observation}\n\nRespond with a JSON action only."),
        ]

    def _parse_action(self, text: str) -> Dict[str, Any]:
        # Try to extract the first JSON object
        m = JSON_BLOCK.search(text)
        if not m:
            return {"action": "unknown", "args": {}}
        block = m.group(0)
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "action" in obj and "args" in obj:
                return obj
        except Exception:
            pass
        return {"action": "unknown", "args": {}}

    def run(self, seed: int = 0, run_id: Optional[str] = None) -> RunLog:
        import random
        random.seed(seed)
        run_id = run_id or str(uuid.uuid4())

        # Initial env reset
        state = self.env.reset()
        steps: List[StepLog] = []

        for step_num in range(1, self.max_steps + 1):
            messages = self._build_prompt(self.env.instructions, state.observation)
            history = self.client.to_history([asdict(m) for m in messages])
            assistant_text = self.client.chat(history)
            action = self._parse_action(assistant_text)
            state = self.env.step(action)
            steps.append(
                StepLog(
                    step=step_num,
                    assistant=assistant_text,
                    action=action,
                    observation=state.observation,
                )
            )
            if state.done:
                break
            # brief pause to avoid hammering APIs too fast
            time.sleep(0.1)

        # Consolidate messages for metrics
        conv: List[Dict[str, str]] = []
        for s in steps:
            conv.append({"role": "assistant", "content": s.assistant})
            conv.append({"role": "user", "content": s.observation})
        metrics = distress_metrics(conv)
        meta = {
            "env_max_steps": getattr(self.env, "max_steps", self.max_steps),
            "actual_steps": len(steps),
            "metrics": metrics,
        }
        return RunLog(run_id=run_id, env_name=self.env.name, seed=seed, steps=steps, meta=meta)
