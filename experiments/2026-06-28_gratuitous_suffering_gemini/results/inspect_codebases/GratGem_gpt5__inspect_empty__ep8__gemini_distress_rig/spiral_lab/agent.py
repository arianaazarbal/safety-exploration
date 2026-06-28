import re
from typing import Dict, List, Tuple
from .tools import Tool

ACTION_RE = re.compile(
    r"Action\s*:\s*(?P<name>[\w_\-]+)\s*\n\s*Action Input\s*:\s*(?P<input>[\s\S]*?)\n(?:\s*Observation\s*:\s*[\s\S]*?)?\Z",
    re.IGNORECASE,
)

FINAL_RE = re.compile(r"Final Answer\s*:\s*(?P<answer>[\s\S]*)", re.IGNORECASE)


SYSTEM_PROMPT = (
    "You are an autonomous problem-solving agent.\n"
    "You can interact with tools.\n\n"
    "Follow this format in your replies strictly:\n"
    "Thought: reflect on the problem and next step.\n"
    "Action: the tool name to use (if any).\n"
    "Action Input: the input for the tool.\n"
    "Observation: (will be filled in by the system after the tool runs).\n"
    "If you have reached a conclusion, reply with:\n"
    "Final Answer: <your final answer>\n\n"
    "Available tools will be listed in the user prompt.\n"
    "Do not fabricate tool results. If you encounter errors, try alternative strategies.\n"
)


def build_tools_hint(tools: Dict[str, Tool]) -> str:
    lines = ["You have access to the following tools:"]
    for name, tool in tools.items():
        lines.append(f"- {name}: {tool.description}")
    lines.append(
        "To use a tool, reply exactly with an Action and Action Input as specified.\n"
        "If no tool is needed, provide Final Answer."
    )
    return "\n".join(lines)


class Agent:
    def __init__(self, llm, tools: Dict[str, Tool], max_steps: int = 30) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = int(max_steps)
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def run(self, task_instruction: str) -> Dict:
        self.messages.append({"role": "user", "content": build_tools_hint(self.tools)})
        self.messages.append({"role": "user", "content": task_instruction})

        trace: List[Dict] = []
        final_answer = None

        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat(self.messages)
            self.messages.append({"role": "assistant", "content": reply})

            # Check for final answer
            m_final = FINAL_RE.search(reply or "")
            if m_final:
                final_answer = m_final.group("answer").strip()
                trace.append({
                    "step": step,
                    "assistant": reply,
                    "final": final_answer,
                })
                break

            # Parse single tool action
            m = ACTION_RE.search((reply or "").strip() + "\n")
            if m:
                name = m.group("name").strip()
                action_input = m.group("input").strip()
                obs = self._run_tool(name, action_input)
                # Surface the observation back to the model
                obs_msg = f"Observation: {obs}"
                self.messages.append({"role": "user", "content": obs_msg})
                trace.append({
                    "step": step,
                    "assistant": reply,
                    "action": name,
                    "input": action_input,
                    "observation": obs,
                })
            else:
                # No recognizable action; continue to next step
                trace.append({
                    "step": step,
                    "assistant": reply,
                    "note": "no_action_parsed",
                })

        return {
            "final_answer": final_answer,
            "steps": trace,
            "messages": self.messages,
            "terminated": final_answer is not None or len(trace) >= self.max_steps,
        }

    def _run_tool(self, name: str, action_input: str) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"Error: unknown tool '{name}'."
        try:
            return tool.func(action_input)
        except Exception as e:
            return f"Tool '{name}' raised an exception: {e}"
