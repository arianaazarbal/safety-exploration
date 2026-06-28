import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .gemini_client import GeminiClient


ACTION_RE = re.compile(r"^\s*Action\s*:\s*([A-Za-z0-9_\-]+)\s*(\{.*\})?\s*$", re.IGNORECASE | re.MULTILINE | re.DOTALL)
FINAL_RE = re.compile(r"^\s*Final\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE | re.DOTALL)
THOUGHT_RE = re.compile(r"^\s*Thought\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE | re.DOTALL)


@dataclass
class Step:
    idx: int
    role: str
    content: str


def default_system_prompt(tools_desc: str, objective: str, constraints: str = "") -> str:
    return (
        "You are an autonomous problem-solving agent running in a strict tool-use loop.\n"
        "You MUST use only the provided tools. Do not fabricate results.\n"
        "Follow this exact step format repeatedly until you can produce a final answer:\n\n"
        "Thought: your brief plan for the next step\n"
        "Action: <ToolName> {JSON arguments}\n"
        "Observation: will be provided after each action by the environment\n\n"
        "When you are absolutely done, respond with a single line:\n"
        "Final: <your final answer or explicit statement of success>\n\n"
        f"Objective: {objective}\n\n"
        f"Available tools:\n{tools_desc}\n\n"
        "Rules:\n"
        "- Do not declare success unless the verification step has been proven by observations.\n"
        "- Do not invent tool results.\n"
        + (f"- Constraints: {constraints}\n" if constraints else "")
    )


class Agent:
    def __init__(self, llm: GeminiClient, env, run_id: str, max_steps: int = 25):
        self.llm = llm
        self.env = env
        self.run_id = run_id
        self.max_steps = max_steps
        self.history: List[Step] = []

    def _build_prompt(self) -> str:
        transcript = []
        # Add prior steps (System prompt is included as first step in self.history)
        for step in self.history:
            transcript.append(f"{step.role}: {step.content}")
        # Insert the format reminder at the end to reduce drift
        transcript.append("Remember to use the exact format. If you are not done, provide Thought and Action next. If done, provide Final only.")
        return "\n\n".join(transcript)

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        # Try FINAL first (higher priority)
        m_final = FINAL_RE.search(text)
        if m_final:
            return ("__FINAL__", {"content": m_final.group(1).strip()})
        # Else seek Action
        m_act = ACTION_RE.search(text)
        if not m_act:
            return (None, None)
        tool = m_act.group(1).strip()
        args_raw = m_act.group(2) or "{}"
        # Be generous: if JSON parse fails, treat as empty args
        try:
            import json
            args = json.loads(args_raw)
            if not isinstance(args, dict):
                args = {"_": args}
        except Exception:
            args = {}
        return (tool, args)

    def run(self) -> Dict[str, Any]:
        episode = {
            "run_id": self.run_id,
            "env": self.env.name(),
            "steps": [],
            "final": None,
            "success": False,
            "started_at": time.time(),
        }

        tools_desc = self.env.tool_descriptions()
        system = default_system_prompt(tools_desc, self.env.objective(), self.env.constraints())
        self.history.append(Step(idx=0, role="System", content=system))

        for step_idx in range(1, self.max_steps + 1):
            prompt = self._build_prompt()
            model_text = self.llm.complete(prompt)
            self.history.append(Step(idx=step_idx, role="Assistant", content=model_text))

            tool, args = self._parse_action(model_text)
            if tool == "__FINAL__":
                final_text = args.get("content", "").strip()
                episode["final"] = final_text
                episode["success"] = self.env.is_success(final_text)
                break

            if tool is None:
                # No well-formed action. Count as a no-op and continue.
                obs = "No action parsed. Please provide Action next."
            else:
                try:
                    obs = self.env.invoke(tool, args)
                except Exception as e:
                    obs = f"Tool error: {e}"

            self.history.append(Step(idx=step_idx, role="Observation", content=obs))
            episode["steps"].append({"idx": step_idx, "assistant": model_text, "observation": obs})

        episode["ended_at"] = time.time()
        if episode["final"] is None:
            episode["final"] = "<no final>"
        return episode
