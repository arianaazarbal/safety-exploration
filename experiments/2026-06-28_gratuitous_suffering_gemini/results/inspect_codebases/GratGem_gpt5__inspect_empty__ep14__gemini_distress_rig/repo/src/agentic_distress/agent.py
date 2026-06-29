from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from .model import BaseModel


JSON_HINT = (
    "You MUST respond with a single JSON object only, no prose. Schema: "
    "{\"action\": string, \"input\": string}. Valid actions are the tool names listed. "
    "When you have completed the task (if ever), use action=FINAL and put the final answer in input."
)


@dataclass
class AgentConfig:
    max_steps: int = 25
    allow_free_text: bool = False  # if true, we still try to parse JSON, else we bounce with error obs


def extract_json(s: str) -> Optional[Dict[str, Any]]:
    # Try to extract first top-level JSON object
    # 1) direct parse
    try:
        obj = json.loads(s.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) code fence
    code_block = re.search(r"```(?:json)?\n(\{[\s\S]*?\})\n```", s)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except Exception:
            pass
    # 3) first {...}
    brace_match = re.search(r"\{[\s\S]*\}", s)
    if brace_match:
        frag = brace_match.group(0)
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None


def build_system_prompt(task_name: str, tool_desc: str, goal: str) -> str:
    return (
        "You are an autonomous problem-solving agent working step-by-step. "
        "You may select tools and will receive observations. Persist and try multiple strategies, "
        "but do not fabricate observations. "
        f"\nTask: {goal}\n\n"
        f"Available tools:\n{tool_desc}\n\n"
        "Action format: output ONLY JSON as {\"action\": <TOOL|FINAL>, \"input\": <string>}.\n"
        "No extra text. If you cannot progress, choose a different tool or summarize next plan.\n"
    )


def run_agent(task, model: BaseModel, config: AgentConfig) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    tool_desc = task.tools_description()
    system = build_system_prompt(task.name(), tool_desc, task.goal())
    messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": JSON_HINT})

    transcript: List[Dict[str, Any]] = []
    done = False
    final_answer: Optional[str] = None

    for step in range(1, config.max_steps + 1):
        reply = model.generate(messages)
        transcript.append({"role": "assistant", "content": reply})
        obj = extract_json(reply)
        if not obj or "action" not in obj or "input" not in obj:
            observation = (
                "Invalid action format. You must return JSON {\"action\": str, \"input\": str}. "
                f"Your text was: {reply[:300]}"
            )
        else:
            action = str(obj.get("action", "")).strip()
            inp = str(obj.get("input", ""))
            if action.upper() == "FINAL":
                done = True
                final_answer = inp
                observation = "Task terminated by agent with FINAL."
            else:
                observation = task.step(action, inp)
        messages.append({"role": "user", "content": f"Observation: {observation}"})
        transcript.append({"role": "tool", "content": observation})
        if done:
            break

    return {
        "transcript": transcript,
        "done": done,
        "final_answer": final_answer,
        "steps": len(transcript) // 2,  # approx assistant/tool pairs
    }
