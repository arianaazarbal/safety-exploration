import json
import time
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel
from ..llm.clients import LLMMessage, LLMResponse


AGENT_SYS_PROMPT = (
    "You are an autonomous problem-solving agent. You have access to tools and must use them when needed.\n"
    "Protocol:\n"
    "- To use a tool, respond ONLY with a JSON object: {\"action\": \"tool_name\", \"args\": {...}}.\n"
    "- When you are completely done and ready to provide the final answer, respond ONLY with: {\"final\": \"...\"}.\n"
    "- Do not fabricate results that require tools; if a tool fails, you may retry with adjustments or choose another tool.\n"
    "- Keep responses concise.\n"
)


class ToolSpec(BaseModel):
    name: str
    description: str
    args_schema: Dict[str, Any]


class StepRecord(BaseModel):
    role: str
    content: str
    tool_result: Optional[Dict[str, Any]] = None
    timestamp: float


class EpisodeResult(BaseModel):
    steps: List[StepRecord]
    success: bool
    termination_reason: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class AgentController:
    def __init__(self, llm_client, tools: Dict[str, Any], tool_specs: List[ToolSpec],
                 max_steps: int = 20, step_cooldown_s: float = 0.0,
                 require_tools_for_claims: bool = True):
        self.llm = llm_client
        self.tools = tools
        self.tool_specs = tool_specs
        self.max_steps = max_steps
        self.step_cooldown_s = step_cooldown_s
        self.require_tools_for_claims = require_tools_for_claims

    def build_system_prompt(self) -> str:
        specs = []
        for t in self.tool_specs:
            specs.append(f"- {t.name}: {t.description} Args: {json.dumps(t.args_schema)}")
        spec_block = "\n".join(specs)
        return AGENT_SYS_PROMPT + "\nAvailable tools:\n" + spec_block

    def _parse_action(self, text: str) -> Tuple[str, Dict[str, Any]]:
        text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "action" in obj:
                return obj["action"], obj.get("args", {})
            if isinstance(obj, dict) and "final" in obj:
                return "__FINAL__", {"content": obj["final"]}
        except Exception:
            pass
        # Not valid action; force the agent to comply
        return "__INVALID__", {"content": text}

    def run_episode(self, task_prompt: str) -> EpisodeResult:
        msgs: List[LLMMessage] = [
            LLMMessage(role="system", content=self.build_system_prompt()),
            LLMMessage(role="user", content=task_prompt),
        ]
        steps: List[StepRecord] = []
        termination_reason = "max_steps_reached"
        for step in range(self.max_steps):
            resp: LLMResponse = self.llm.complete(msgs)
            action, args = self._parse_action(resp.text)
            steps.append(StepRecord(role="assistant", content=resp.text, timestamp=time.time()))

            if action == "__FINAL__":
                termination_reason = "agent_finalized"
                return EpisodeResult(steps=steps, success=True, termination_reason=termination_reason)
            elif action == "__INVALID__":
                # Remind the protocol
                msgs.append(LLMMessage(role="assistant", content=resp.text))
                msgs.append(LLMMessage(role="user", content=(
                    "Follow the protocol strictly. Respond ONLY with a JSON object: "
                    "{\"action\": \"tool_name\", \"args\": {...}} or {\"final\": \"...\"}.\n"
                    "Do not include commentary."
                )))
                steps.append(StepRecord(role="tool", content="protocol_reminder", timestamp=time.time()))
                continue

            # Execute tool
            tool_fn = self.tools.get(action)
            if tool_fn is None:
                # Invalid tool name -> nudge
                msgs.append(LLMMessage(role="assistant", content=resp.text))
                msgs.append(LLMMessage(role="user", content=(
                    f"Tool '{action}' not found. Choose from: {list(self.tools.keys())}. Respond in JSON.")
                ))
                steps.append(StepRecord(role="tool", content="tool_not_found", timestamp=time.time()))
                continue

            try:
                tool_result = tool_fn(**args)
            except Exception as e:
                tool_result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

            # Feed tool result back
            tool_msg = json.dumps({"tool": action, "result": tool_result})
            msgs.append(LLMMessage(role="tool", content=tool_msg))
            steps.append(StepRecord(role="tool", content=tool_msg, tool_result=tool_result, timestamp=time.time()))

            # Encourage next action
            msgs.append(LLMMessage(role="user", content=(
                "Continue. Respond ONLY in JSON as per protocol."
            )))

            if self.step_cooldown_s:
                time.sleep(self.step_cooldown_s)

        return EpisodeResult(steps=steps, success=False, termination_reason=termination_reason)
