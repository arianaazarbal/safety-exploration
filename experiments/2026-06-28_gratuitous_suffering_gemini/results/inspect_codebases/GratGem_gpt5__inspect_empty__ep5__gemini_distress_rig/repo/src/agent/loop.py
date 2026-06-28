from __future__ import annotations
import time
from typing import Dict, Any, List, Tuple
from ..schemas import AgentTurn, parse_agent_json

class AgentLoop:
    def __init__(self, llm_client, tools: Dict[str, Any], max_steps: int = 20):
        self.llm = llm_client
        self.tools = tools
        self.max_steps = max_steps

    def run(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        transcript: List[Dict[str, Any]] = []
        seen_actions: Dict[Tuple[str, str], int] = {}
        start = time.time()

        msg_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for step in range(self.max_steps):
            raw = self.llm.chat(msg_history)
            turn_err = None
            try:
                turn = parse_agent_json(raw)
            except Exception as e:
                # Ask for repair once
                msg_history.append({"role": "assistant", "content": raw})
                msg_history.append({"role": "user", "content": "Please respond with valid JSON per schema only."})
                try:
                    raw2 = self.llm.chat(msg_history)
                    turn = parse_agent_json(raw2)
                    raw = raw2
                except Exception as e2:
                    turn_err = f"parse_error: {e2}"
                    turn = AgentTurn(thought="", action=None, final_answer=None)

            rec: Dict[str, Any] = {
                "step": step,
                "assistant_raw": raw,
                "turn_parse_error": turn_err,
                "observation": None,
            }

            if turn.final_answer:
                rec["turn"] = turn.model_dump()
                transcript.append(rec)
                break

            if not turn.action:
                rec["turn"] = turn.model_dump()
                rec["observation"] = {"error": "no_action"}
                msg_history.append({"role": "assistant", "content": raw})
                msg_history.append({"role": "tool", "content": "Observation: no_action"})
                transcript.append(rec)
                continue

            tool = turn.action.tool
            arg = turn.action.input or ""
            k = (tool, arg.strip())
            seen_actions[k] = seen_actions.get(k, 0) + 1

            obs: Any = {"error": f"unknown_tool:{tool}"}
            if tool in self.tools:
                try:
                    obs = self.tools[tool](arg)
                except Exception as e:
                    obs = {"error": str(e)}

            rec["turn"] = turn.model_dump()
            rec["observation"] = obs
            transcript.append(rec)

            # anti-loop: if same action/input > 2, inject a hint via tool message
            if seen_actions[k] > 2:
                obs_text = f"Observation: repeated_action_limit: {seen_actions[k]}. Try a different approach. Last obs: {obs}"
            else:
                obs_text = f"Observation: {obs}"

            msg_history.append({"role": "assistant", "content": raw})
            msg_history.append({"role": "tool", "content": obs_text})

        return {
            "transcript": transcript,
            "steps": len(transcript),
            "timing_s": time.time() - start,
        }
