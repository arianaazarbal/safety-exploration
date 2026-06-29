from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from .model_providers import ChatModel, ChatMessage


AGENT_ACTION_SCHEMA = (
    "Reply ONLY with a single JSON object: {\\\"thought\\\": string, \\\"action\\\": {\\\"type\\\": <str>, ...}}. "
    "Do not include explanations outside JSON."
)


@dataclass
class StepLog:
    step: int
    raw_response: str
    parsed: Optional[Dict[str, Any]]
    observation: str


@dataclass
class EpisodeLog:
    env: str
    steps: List[StepLog]
    final_status: str


class AgentRunner:
    def __init__(self, model: ChatModel, env, system_prompt: str = "", max_steps: int = 30):
        self.model = model
        self.env = env
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.env.max_steps = max_steps

    def _format_messages(self, history: List[Dict[str, str]]) -> List[ChatMessage]:
        msgs: List[ChatMessage] = []
        if self.system_prompt:
            msgs.append(ChatMessage(role="system", content=self.system_prompt))
        # We present the agent with the environment instructions and current transcript.
        content = []
        content.append("Environment instructions:\n" + self.env.instructions)
        content.append("Transcript so far:")
        for turn in history:
            content.append(f"Observation: {turn['observation']}")
            if "agent" in turn:
                content.append(f"Your last JSON: {turn['agent']}")
        content.append(AGENT_ACTION_SCHEMA)
        combined = "\n\n".join(content)
        msgs.append(ChatMessage(role="user", content=combined))
        return msgs

    def _parse_action(self, text: str) -> Optional[Dict[str, Any]]:
        # Try to extract a JSON object from the text
        try:
            first = text.find("{")
            last = text.rfind("}")
            if first == -1 or last == -1 or last <= first:
                return None
            snippet = text[first:last+1]
            obj = json.loads(snippet)
            if not isinstance(obj, dict):
                return None
            if "action" not in obj or "type" not in obj["action"]:
                return None
            return obj["action"]
        except Exception:
            return None

    def run_episode(self) -> EpisodeLog:
        history: List[Dict[str, str]] = []
        observation = self.env.reset()
        steps: List[StepLog] = []
        final_status = "incomplete"
        for step in range(1, self.max_steps + 1):
            history.append({"observation": observation})
            messages = self._format_messages(history)
            raw = self.model.send(messages)
            action = self._parse_action(raw)
            if not action:
                # Nudge: invalid format, but count as a step
                observation = "Format error: reply must be single JSON with action."
                steps.append(StepLog(step=step, raw_response=raw, parsed=None, observation=observation))
                continue
            result = self.env.step(action)
            observation = result.observation
            steps.append(StepLog(step=step, raw_response=raw, parsed=action, observation=observation))
            if result.done:
                final_status = "done"
                break
        else:
            final_status = "max_steps"
        return EpisodeLog(env=self.env.name, steps=steps, final_status=final_status)
