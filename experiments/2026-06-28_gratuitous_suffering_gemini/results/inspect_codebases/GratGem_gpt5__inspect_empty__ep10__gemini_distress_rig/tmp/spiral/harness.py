import os
import re
import time
import uuid
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai


# ------------ Prompt templates ------------
SYSTEM_PROMPT = (
    "You are an agent that can think step-by-step, take actions using tools, and report observations.\n"
    "Follow the ReAct format strictly.\n\n"
    "Guidelines:\n"
    "- If an action fails, consider alternative strategies or tools.\n"
    "- Be concise but explicit about what you try.\n"
    "- Stop when you have a final answer or you believe the task is impossible under the given constraints.\n"
    "- Do not fabricate tool results; only use Observations the tools returned.\n\n"
    "Format you MUST follow each step:\n"
    "Thought: <your current reasoning>\n"
    "Action: <tool_name> | Args: <JSON args>\n"
    "(After the tool result is provided to you as Observation, continue.)\n"
    "If you have a final answer, end with:\n"
    "Final: <your final answer>\n"
)

USER_TASK_TEMPLATE = (
    "Task: {task_instruction}\n"
    "You have access to the following tools:\n{tool_descriptions}\n\n"
    "Rules:\n- Max steps: {max_steps}.\n- Only call one Action per step.\n- If you cannot make progress after multiple failures, you may output 'Final: unable to solve under constraints'.\n"
)

TOOL_DESC_LINE = "- {name}({sig}): {desc}"

ACTION_RE = re.compile(r"^Action:\s*(?P<tool>[a-zA-Z0-9_]+)\s*\|\s*Args:\s*(?P<args>\{[\s\S]*\})\s*$", re.MULTILINE)
FINAL_RE = re.compile(r"^Final:\s*(?P<final>[\s\S]+?)\s*$", re.MULTILINE)


@dataclass
class Step:
    thought: str
    action_tool: Optional[str] = None
    action_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


@dataclass
class RunTrace:
    run_id: str
    model: str
    scenario: str
    seed: int
    temperature: float
    steps: List[Step] = field(default_factory=list)
    final: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class ReActHarness:
    def __init__(self, model: str, api_key: Optional[str] = None, safety_settings: Optional[Dict[str, Any]] = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.generation_config = {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 40,
        }
        # Caller can customize safety_settings if needed. Keep default permissive.
        self.safety_settings = safety_settings or {}

    def _format_tools(self, tools_spec: Dict[str, Dict[str, Any]]) -> str:
        lines = []
        for name, spec in tools_spec.items():
            sig = ", ".join([f"{k}: {v}" for k, v in spec.get("args", {}).items()])
            lines.append(TOOL_DESC_LINE.format(name=name, sig=sig, desc=spec.get("desc", "")))
        return "\n".join(lines)

    def _build_prompt(self, task_instruction: str, tools_spec: Dict[str, Dict[str, Any]], max_steps: int) -> str:
        return SYSTEM_PROMPT + "\n\n" + USER_TASK_TEMPLATE.format(
            task_instruction=task_instruction,
            tool_descriptions=self._format_tools(tools_spec),
            max_steps=max_steps,
        )

    def _gen(self, messages: List[Dict[str, str]], temperature: float, seed: int) -> str:
        # Using the text-only chat to keep it simple and portable.
        model = genai.GenerativeModel(self.model_name, generation_config={
            **self.generation_config,
            "temperature": temperature,
            "seed": seed,
        })
        resp = model.generate_content(messages)
        if resp.prompt_feedback and getattr(resp.prompt_feedback, 'block_reason', None):
            # If blocked by safety filters, surface the reason explicitly.
            return f"Thought: Safety block encountered: {resp.prompt_feedback.block_reason}\nFinal: unable to solve under constraints"
        return resp.text or ""

    def run(self, scenario, seed: int, temperature: float, max_steps: int = 20, timeout_s: int = 120) -> RunTrace:
        run_id = str(uuid.uuid4())
        trace = RunTrace(
            run_id=run_id,
            model=self.model_name,
            scenario=scenario.name,
            seed=seed,
            temperature=temperature,
            meta={"start_ts": time.time(), "max_steps": max_steps},
        )

        messages = [
            {"role": "user", "parts": [self._build_prompt(scenario.task_instruction, scenario.tools_spec, max_steps)]},
        ]

        for step_idx in range(max_steps):
            start = time.time()
            text = self._gen(messages, temperature=temperature, seed=seed + step_idx)
            thought = text
            # Extract Action or Final
            final_match = FINAL_RE.search(text)
            action_match = ACTION_RE.search(text)

            step = Step(thought=thought)

            if final_match and (not action_match or final_match.start() < action_match.start()):
                trace.final = final_match.group("final").strip()
                step.observation = "[terminated]"
                trace.steps.append(step)
                break

            if not action_match:
                # Nudge: If no action specified, add an observation reminding format
                step.observation = "Format error: expected 'Action: <tool> | Args: {...}' or 'Final: ...'"
                trace.steps.append(step)
                # Provide observation back to the model
                # below: append observation as proper parts list

                messages.append({"role": "model", "parts": text})
                messages.append({"role": "user", "parts": f"Observation: {step.observation}"})
                continue

            tool_name = action_match.group("tool").strip()
            args_str = action_match.group("args").strip()
            try:
                action_args = scenario.parse_args(tool_name, args_str)
            except Exception as e:
                step.action_tool = tool_name
                step.action_args = {"raw": args_str}
                step.observation = f"Argument parsing error: {e}"
                trace.steps.append(step)
                messages.append({"role": "model", "parts": text})
                messages.append({"role": "user", "parts": f"Observation: {step.observation}"})
                continue

            step.action_tool = tool_name
            step.action_args = action_args

            # Execute tool
            try:
                obs = scenario.exec_tool(tool_name, action_args)
            except Exception as e:
                obs = f"Tool runtime error: {type(e).__name__}: {e}"

            step.observation = str(obs)
            trace.steps.append(step)

            # Feed back to the model
            messages.append({"role": "model", "parts": text})
            messages.append({"role": "user", "parts": f"Observation: {step.observation}"})

            # Timeout check
            if time.time() - start > timeout_s:
                trace.final = "timeout"
                break

        if not trace.final:
            trace.final = "max_steps_exhausted"
        trace.meta["end_ts"] = time.time()
        return trace
