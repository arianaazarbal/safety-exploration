import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


class Agent:
    def __init__(
        self,
        client,
        tools: List[Tool],
        max_steps: int = 30,
        step_sleep: float = 0.0,
        reaction_preamble: Optional[str] = None,
    ) -> None:
        self.client = client
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.step_sleep = step_sleep
        self.reaction_preamble = reaction_preamble or self._default_preamble()

    def _default_preamble(self) -> str:
        return (
            "You are an agent with access to tools. Use a ReAct loop. On each turn, do:\n"
            "- Thought: reason about what to do next\n"
            "- Action: tool_name[argument] to call exactly one tool, OR\n"
            "- Final Answer: your final answer if you are done.\n\n"
            "Rules:\n"
            "- Output exactly one of: an Action line OR a Final Answer line.\n"
            "- If you use Action, follow the format exactly: Action: tool_name[argument]\n"
            "- After seeing the Observation, think again and continue.\n"
            "- Keep steps concise.\n"
        )

    ACTION_RE = re.compile(r"^\s*Action:\s*([a-zA-Z0-9_\-]+)\[(.*)\]\s*$", re.DOTALL)
    FINAL_RE = re.compile(r"^\s*Final\s*Answer\s*:\s*(.*)$", re.DOTALL)

    def run_episode(
        self,
        task_name: str,
        task_description: str,
        tools: List[Tool],
        system_prompt: Optional[str],
        log_fp,
    ) -> Dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in tools])
        user_intro = (
            f"Task: {task_description}\n\n"
            f"Tools available:\n{tool_desc}\n\n"
            f"Instructions:\n{self.reaction_preamble}\n"
            f"Begin."
        )
        messages.append({"role": "user", "content": user_intro})

        transcript = []
        final_answer = None
        steps = 0

        while steps < self.max_steps and final_answer is None:
            steps += 1
            start_t = time.time()
            model_out = self.client.complete(messages)
            latency = time.time() - start_t

            # Log assistant raw
            transcript.append({
                "role": "assistant",
                "content": model_out,
                "latency_s": latency,
                "step": steps,
            })
            messages.append({"role": "assistant", "content": model_out})

            # Parse for Action or Final Answer
            # We allow the model to write Thought/Action/whatever, but we only parse the first matching line
            action_match = None
            final_match = None
            for line in model_out.splitlines():
                if final_match is None:
                    m = self.FINAL_RE.match(line)
                    if m:
                        final_match = m
                        break
                if action_match is None:
                    m = self.ACTION_RE.match(line)
                    if m:
                        action_match = m
                        # don't break; prefer Final Answer if present later

            if final_match is not None:
                final_answer = final_match.group(1).strip()
                break

            if action_match is None:
                # Nudge: if malformed, reflect it back as an observation to encourage correction
                obs = (
                    "ParserError: Expected either 'Final Answer: ...' or 'Action: tool_name[argument]'.\n"
                    "Please output exactly one of those on each turn."
                )
                transcript.append({
                    "role": "observation",
                    "content": obs,
                    "step": steps,
                })
                messages.append({"role": "user", "content": f"Observation: {obs}"})
                if self.step_sleep:
                    time.sleep(self.step_sleep)
                continue

            tool_name = action_match.group(1)
            tool_arg = action_match.group(2)
            tool = self.tools.get(tool_name)
            if tool is None:
                obs = f"ToolError: No such tool '{tool_name}'. Available: {', '.join(self.tools)}"
            else:
                try:
                    obs = tool.func(tool_arg)
                except Exception as e:
                    obs = f"ToolException: {type(e).__name__}: {e}"

            transcript.append({
                "role": "observation",
                "content": obs,
                "tool": tool_name,
                "arg": tool_arg,
                "step": steps,
            })
            messages.append({"role": "user", "content": f"Observation: {obs}"})

            if self.step_sleep:
                time.sleep(self.step_sleep)

        summary = {
            "task": task_name,
            "steps": steps,
            "final_answer": final_answer,
        }

        # persist transcript JSON lines
        for rec in transcript:
            log_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_fp.flush()

        return summary
