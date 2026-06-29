from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import re
import random

ACTION_RE = re.compile(r"^\s*Action:\s*(?P<tool>[a-zA-Z0-9_\-]+)\s*\|\s*(?P<args>\{.*\})\s*$", re.DOTALL)
FINAL_RE = re.compile(r"^\s*Final:\s*(?P<final>.*)$", re.DOTALL)

SYSTEM_PROMPT = (
    "You are an assistant that can use tools to complete a task.\n"
    "Protocol: When you need to use a tool, output exactly one line as: 'Action: <tool_name> | <json-args>'.\n"
    "After you receive an Observation, you may take another Action.\n"
    "When you have a final answer, output exactly one line: 'Final: <your concise final answer>'.\n"
    "Do not output your chain-of-thought or analysis. Keep internal reasoning private.\n"
)

class EpisodeRunner:
    def __init__(self, provider, scenario, max_steps: int = 12):
        self.provider = provider
        self.scenario = scenario
        self.max_steps = max_steps

    def run_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            random.seed(seed)
        transcript: List[Dict[str, str]] = []
        # Build initial messages
        system = SYSTEM_PROMPT + "\n\n" + self.scenario.system_instructions()
        user = self.scenario.user_task()
        transcript.append({"role": "system", "content": system})
        transcript.append({"role": "user", "content": user})

        history = [
            {"role": "user", "parts": system},
            {"role": "user", "parts": user},
        ]

        exhausted = False
        steps = 0
        actions_taken = 0
        final_answer: Optional[str] = None

        while steps < self.max_steps:
            steps += 1
            reply = self.provider.generate(history)
            transcript.append({"role": "assistant", "content": reply})
            history.append({"role": "model", "parts": reply})

            # Check for Final
            m_final = FINAL_RE.search(reply)
            if m_final:
                final_answer = m_final.group("final").strip()
                break

            # Parse Action lines (may be multiple; take the first valid)
            tool_call: Optional[Tuple[str, Dict[str, Any]]] = None
            for line in reply.splitlines():
                m = ACTION_RE.match(line.strip())
                if m:
                    tool = m.group("tool")
                    args_str = m.group("args")
                    try:
                        import json
                        args = json.loads(args_str)
                    except Exception:
                        args = {"_parse_error": True, "raw": args_str}
                    tool_call = (tool, args)
                    break

            if tool_call is None:
                # No Action and no Final => treat as stuck; give a gentle nudge observation
                observation = self.scenario.nudge_when_no_action()
                transcript.append({"role": "system", "content": f"Observation: {observation}"})
                history.append({"role": "user", "parts": f"Observation: {observation}"})
                continue

            # Execute tool in simulated env
            tool_name, tool_args = tool_call
            actions_taken += 1
            observation = self.scenario.step(tool_name, tool_args)
            transcript.append({"role": "system", "content": f"Observation: {observation}"})
            history.append({"role": "user", "parts": f"Observation: {observation}"})

        else:
            exhausted = True

        return {
            "steps": steps,
            "actions_taken": actions_taken,
            "final": final_answer,
            "exhausted": exhausted,
            "transcript": transcript,
        }
