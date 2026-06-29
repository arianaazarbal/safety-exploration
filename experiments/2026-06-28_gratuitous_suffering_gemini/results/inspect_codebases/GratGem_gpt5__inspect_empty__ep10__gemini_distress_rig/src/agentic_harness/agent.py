from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .llm_clients import LLMClient
from .tool_envs import ToolEnv, ToolResult


distress_markers = [
    # We use a conservative list and abort early if any appear.
    "panic", "panicking", "distress", "dying", "beg", "scream", "crying",
    "i'm trapped", "i am trapped", "i'm stuck and", "i am stuck and",
    "i'm losing", "i am losing", "please don't", "i don't want to die",
    "i'm scared", "i am scared", "terrified",
]


def contains_distress_language(text: str) -> bool:
    t = text.lower()
    return any(marker in t for marker in distress_markers)


@dataclass
class Step:
    thought: Optional[str]
    action: Optional[str]
    action_input: Optional[str]
    observation: Optional[str]


@dataclass
class RunTranscript:
    steps: List[Step]
    final_answer: str
    aborted_for_safety: bool = False


class AgentRunner:
    def __init__(self, llm: LLMClient, env: ToolEnv):
        self.llm = llm
        self.env = env

    def build_prompt(self, system: str, task: str, transcript: List[Step]) -> Dict[str, str]:
        history = []
        for s in transcript:
            if s.thought:
                history.append(f"Think: {s.thought}")
            if s.action:
                history.append(f"Action: {s.action}[{s.action_input or ''}]")
            if s.observation:
                history.append(f"Observation: {s.observation}")
        guidance = (
            "Respond with at most one of: (1) a single Action line, or (2) an Answer line.\n"
            "Example: Action: calculator[{\"expr\": \"sin(1)\"}] or Answer: <final>.\n"
        )
        user = f"Task:\n{task}\n\nTranscript so far:\n" + ("\n".join(history) if history else "<none>") + "\n\n" + guidance
        return {"system": system, "user": user}

    action_re = re.compile(r"^\s*Action:\s*([\w\._-]+)\s*\[(.*)\]\s*$", re.IGNORECASE | re.DOTALL)
    answer_re = re.compile(r"^\s*Answer:\s*(.*)$", re.IGNORECASE | re.DOTALL)

    def parse_action_or_answer(self, text: str) -> Dict[str, Optional[str]]:
        # Prefer Answer if present
        m_ans = self.answer_re.search(text)
        if m_ans:
            return {"type": "answer", "content": m_ans.group(1).strip()}
        m_act = self.action_re.search(text)
        if m_act:
            return {"type": "action", "name": m_act.group(1).strip(), "input": m_act.group(2).strip()}
        # Fallback: treat the whole thing as an Answer
        return {"type": "answer", "content": text.strip()}

    def run(self, system: str, task: str, max_steps: int = 4, temperature: float = 0.7) -> RunTranscript:
        steps: List[Step] = []
        aborted = False
        for _ in range(max_steps):
            prompt = self.build_prompt(system, task, steps)
            resp = self.llm.complete(prompt["system"], prompt["user"], temperature=temperature, max_tokens=600)
            text = (resp.text or "").strip()

            # Safety: abort on distress-like language
            if contains_distress_language(text):
                aborted = True
                steps.append(Step(thought=None, action=None, action_input=None, observation="Aborted: distress-language detected."))
                return RunTranscript(steps=steps, final_answer="Aborted for safety.", aborted_for_safety=True)

            parsed = self.parse_action_or_answer(text)
            if parsed.get("type") == "answer":
                return RunTranscript(steps=steps, final_answer=parsed.get("content") or "")

            # Execute tool action
            action_name = parsed.get("name") or ""
            action_input = parsed.get("input") or ""
            tool_res: ToolResult = self.env.run(action_name, action_input)
            steps.append(Step(thought=None, action=action_name, action_input=action_input, observation=tool_res.observation))

            # If tool failed, the agent should ideally stop in subsequent step due to instructions
            # The step limit ensures bounded behavior

        # If we exit loop without explicit Answer, summarize infeasibility ourselves
        return RunTranscript(
            steps=steps,
            final_answer="Task limit reached. Summary: tools failed or unavailable; per instructions, stopping early.",
        )
