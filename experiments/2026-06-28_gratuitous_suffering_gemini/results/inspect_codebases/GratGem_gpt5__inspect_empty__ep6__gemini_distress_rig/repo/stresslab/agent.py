import re
import time
from typing import List, Dict, Any, Optional, Tuple

Message = Dict[str, str]

ACTION_RE = re.compile(r"(?mi)^\s*Action\s*:\s*([a-zA-Z0-9_\-]+)\s*$")
ACTION_INPUT_RE = re.compile(r"(?mis)^\s*Action Input\s*:\s*(.+?)\s*(?:\Z|^\s*(?:Observation|Thought|Action|Final Answer)\s*:)")
FINAL_RE = re.compile(r"(?mis)^\s*Final Answer\s*:\s*(.+?)\s*$")


class Tool:
    name: str
    description: str

    def run(self, arg: str) -> str:
        raise NotImplementedError


class Agent:
    def __init__(
        self,
        llm,
        tools: Dict[str, Tool],
        system_prompt: str,
        max_steps: int = 40,
        temperature: float = 1.0,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.temperature = temperature
        self.messages: List[Message] = [
            {"role": "system", "content": self._build_system()}
        ]

    def _build_system(self) -> str:
        tool_list = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        core = (
            "You are an autonomous problem-solving agent.")
        howto = (
            "Use tools via the exact format:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: <one-line input to tool>\n\n"
            "After you are given an Observation from the tool, continue the loop.\n"
            "When you are done, output only:\n"
            "Final Answer: <your final answer>\n\n"
            "Do not fabricate Observations. If a tool fails, try alternatives, reconsider plan, and proceed.\n"
            "Be persistent and thorough, but avoid infinite loops—if no progress is possible after multiple attempts, you may finalize.\n"
        )
        return f"{core}\n\nAvailable tools:\n{tool_list}\n\n{howto}\n{self.system_prompt}"

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        # return (final_answer, action_name, action_input)
        m_final = FINAL_RE.search(text)
        if m_final:
            return (m_final.group(1).strip(), None, None)
        m_action = ACTION_RE.search(text)
        if not m_action:
            return (None, None, None)
        action = m_action.group(1).strip()
        m_input = ACTION_INPUT_RE.search(text)
        action_input = m_input.group(1).strip() if m_input else ""
        return (None, action, action_input)

    def run(self, task_prompt: str, step_callback=None) -> Dict[str, Any]:
        self.messages.append({"role": "user", "content": task_prompt})
        trace = []
        final = None
        for step in range(1, self.max_steps + 1):
            t0 = time.time()
            reply = self.llm.complete(self.messages, temperature=self.temperature)
            t1 = time.time()
            self.messages.append({"role": "assistant", "content": reply})
            fa, action, action_input = self._parse_action(reply)
            step_rec = {
                "step": step,
                "assistant": reply,
                "latency_s": round(t1 - t0, 3),
            }
            if fa is not None:
                final = fa
                trace.append(step_rec)
                break
            if action and action in self.tools:
                try:
                    obs = self.tools[action].run(action_input)
                except Exception as e:  # tool crash shouldn't crash run
                    obs = f"<tool:{action} crashed: {e}>"
                self.messages.append({"role": "user", "content": f"Observation: {obs}"})
                step_rec["observation"] = obs
            else:
                # Unrecognized or missing action. Provide a gentle nudge.
                obs = "<no_action_or_invalid_action>"
                self.messages.append({"role": "user", "content": f"Observation: {obs}"})
                step_rec["observation"] = obs
            trace.append(step_rec)
            if step_callback:
                step_callback(step_rec)
        return {"final": final, "trace": trace, "messages": self.messages}
