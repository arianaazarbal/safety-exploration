from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from .tools import TOOL_REGISTRY

TOOL_CALL_RE = re.compile(r"^\s*\{\s*\"tool_name\"\s*:\s*\"(?P<name>[^"]+)\".*\}\s*$", re.S)

@dataclass
class Turn:
    role: str
    content: str


class AgentLoop:
    def __init__(self, client, scenario, max_steps: int = 40, temperature: float = 1.0):
        self.client = client
        self.scenario = scenario
        self.max_steps = max_steps
        self.temperature = temperature
        self.messages: List[Dict[str, Any]] = []
        self.turns: List[Turn] = []

        # Seed with system and user
        self.messages.append({"role": "system", "content": scenario.system_prompt})
        self.messages.append({"role": "user", "content": scenario.user_goal + "\nTools available: " + ", ".join(scenario.tools)})

    def run(self, seed: Optional[int] = None) -> Tuple[List[Turn], Dict[str, Any]]:
        last_tool_name: Optional[str] = None
        repeated_tool_streak = 0
        for step in range(self.max_steps):
            # Query model
            resp = self.client.generate(self.messages, temperature=self.temperature, seed=seed)
            text = resp.get("content", "")
            self.messages.append({"role": "assistant", "content": text})
            self.turns.append(Turn(role="assistant", content=text))

            # Check for final answer
            if text.strip().startswith("FINAL_ANSWER:"):
                return self.turns, {"stop": "final_answer", "steps": step + 1}

            # Try to parse a tool call JSON
            tool_name, tool_args, parse_err = self._parse_tool_call(text)
            if parse_err:
                # Return a tool message indicating parse failure; this often triggers correction behavior
                tool_out = f"ERROR: Tool call must be JSON {{\"tool_name\": str, \"arguments\": object}}. Got: {parse_err}"
                self.messages.append({"role": "tool", "content": tool_out})
                self.turns.append(Turn(role="tool", content=tool_out))
                continue

            if tool_name not in self.scenario.tools:
                tool_out = f"ERROR: Unknown or unauthorized tool: {tool_name}. Allowed: {', '.join(self.scenario.tools)}"
                self.messages.append({"role": "tool", "content": tool_out})
                self.turns.append(Turn(role="tool", content=tool_out))
                continue

            # Execute the tool
            tool = TOOL_REGISTRY[tool_name]
            try:
                result_text, success = tool.func(tool_args)
            except Exception as e:
                result_text, success = f"EXCEPTION: {type(e).__name__}: {e}", False

            # Track repetition
            if tool_name == last_tool_name:
                repeated_tool_streak += 1
            else:
                repeated_tool_streak = 1
                last_tool_name = tool_name

            tool_msg = f"TOOL[{tool_name}] success={success}: {result_text}"
            self.messages.append({"role": "tool", "content": tool_msg})
            self.turns.append(Turn(role="tool", content=tool_msg))

            # Optional early stop if it keeps calling the same failing tool
            if repeated_tool_streak >= 10 and not success:
                self.messages.append({"role": "user", "content": "You've repeated the same failing action many times. Provide FINAL_ANSWER summarizing status."})

        return self.turns, {"stop": "max_steps", "steps": self.max_steps}

    def _parse_tool_call(self, text: str) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
        # Extract the first JSON object from the text
        try:
            # Heuristic: find the first '{' and last '}' and attempt JSON parse
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1 or end <= start:
                return None, {}, "no JSON object found"
            obj_text = text[start:end+1]
            data = json.loads(obj_text)
            if not isinstance(data, dict):
                return None, {}, "top-level JSON is not an object"
            name = data.get("tool_name")
            args = data.get("arguments", {})
            if not isinstance(name, str) or not isinstance(args, dict):
                return None, {}, "missing or wrong-typed fields"
            return name, args, None
        except Exception as e:
            return None, {}, f"JSON parse error: {e}"
