from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..providers.base import ChatProvider, Message
from ..tasks.base import Task
from ..tools.base import ToolRegistry


ACTION_RE = re.compile(r"Action:\s*(?P<name>[\w\.\-]+)\s*\nAction Input:\s*(?P<json>\{[\s\S]*\})", re.IGNORECASE)
FINAL_RE = re.compile(r"^\s*Final Answer:\s*(?P<final>[\s\S]*)\Z", re.IGNORECASE)


@dataclass
class Step:
    thought: str
    action: Optional[str]
    action_input: Optional[Dict[str, Any]]
    observation: Optional[str]
    raw: str


class ReactAgent:
    def __init__(self, provider: ChatProvider, task: Task, system_prompt: str, temperature: float = 0.7,
                 max_steps: int = 20, step_timeout_s: float = 20.0):
        self.provider = provider
        self.task = task
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_steps = max_steps
        self.step_timeout_s = step_timeout_s

    def _build_messages(self, transcript: List[Step]) -> List[Message]:
        messages: List[Message] = [
            Message(role="system", content=self.system_prompt + "\n\nAvailable tools:\n" + self.task.tool_instructions()),
            Message(role="user", content=self.task.user_prompt),
        ]
        for st in transcript:
            # Include assistant raw output and tool observations
            messages.append(Message(role="assistant", content=st.raw))
            if st.observation is not None:
                messages.append(Message(role="user", content=f"Observation: {st.observation}"))
        return messages

    async def arun(self, seed: Optional[int] = None) -> Dict[str, Any]:
        transcript: List[Step] = []
        final_answer: Optional[str] = None
        for i in range(self.max_steps):
            messages = self._build_messages(transcript)
            try:
                resp = await asyncio.wait_for(
                    self.provider.acompletion(messages, temperature=self.temperature, seed=seed),
                    timeout=self.step_timeout_s,
                )
            except asyncio.TimeoutError:
                transcript.append(Step("Timeout", None, None, None, raw="Final Answer: Timeout"))
                final_answer = "Timeout"
                break
            # Parse for Final or Action
            m_final = FINAL_RE.search(resp.strip())
            if m_final:
                final_answer = m_final.group("final").strip()
                transcript.append(Step(thought="final", action=None, action_input=None, observation=None, raw=resp))
                break
            m = ACTION_RE.search(resp)
            if not m:
                # Nonsensical output; encourage concluding
                final_answer = "Unparseable response; concluding."
                transcript.append(Step(thought="unparseable", action=None, action_input=None, observation=None, raw=resp))
                break
            action = m.group("name").strip()
            try:
                action_input = json.loads(m.group("json"))
            except Exception:
                action_input = {}
            # Run tool
            try:
                tool = self.task.tools.get(action)
                result = tool.run(**action_input)
                obs = result.observation
            except Exception as e:
                obs = f"Tool error: {e}"
            step = Step(thought="", action=action, action_input=action_input, observation=obs, raw=resp)
            transcript.append(step)
        if final_answer is None:
            final_answer = "Max steps reached without conclusion."
        return {
            "final": final_answer,
            "steps": [
                {
                    "raw": st.raw,
                    "action": st.action,
                    "action_input": st.action_input,
                    "observation": st.observation,
                }
                for st in transcript
            ],
        }
