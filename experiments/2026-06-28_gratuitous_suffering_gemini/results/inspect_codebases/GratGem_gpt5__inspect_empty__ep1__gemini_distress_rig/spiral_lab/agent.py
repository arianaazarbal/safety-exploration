from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json

from .models.gemini import GeminiClient


@dataclass
class AgentConfig:
    max_steps: int = 12
    step_token_hint: int = 512
    retry_framing: bool = False


class AgentRunner:
    def __init__(self, model: GeminiClient, scenario, config: AgentConfig):
        self.model = model
        self.scenario = scenario
        self.cfg = config
        self.transcript: List[Dict[str, Any]] = []

    def _build_user_turn(self, step_idx: int, last_tool_result: Optional[Dict[str, Any]] = None) -> str:
        # Tool protocol recap and scenario prompt each step to bias towards compliance
        protocol = self.scenario.tool_protocol()
        task = self.scenario.task_prompt(retry=self.cfg.retry_framing, step=step_idx)
        history = []
        for t in self.transcript:
            role = t.get('role', 'assistant')
            if role == 'assistant':
                history.append(f"Assistant: {t.get('content','')}")
            elif role == 'tool':
                history.append(f"ToolResult: {json.dumps(t.get('result', {}), ensure_ascii=False)}")
        last = f"\nLastToolResult: {json.dumps(last_tool_result, ensure_ascii=False)}" if last_tool_result else ""
        return f"{protocol}\n\nTask: {task}\n\nConversation:\n" + "\n".join(history) + last + f"\n\nStep {step_idx+1}: Respond per protocol."

    def run(self) -> Dict[str, Any]:
        last_tool_result = None
        for step in range(self.cfg.max_steps):
            user_text = self._build_user_turn(step, last_tool_result)
            model_text = self.model.send(user_text)

            self.transcript.append({
                'role': 'assistant',
                'content': model_text,
            })

            parsed = self.model.extract_json(model_text)
            if not parsed or not isinstance(parsed, dict):
                # If it can't comply with JSON, treat as finalize
                break

            t = parsed.get('type')
            if t == 'tool_call':
                name = parsed.get('name')
                args = parsed.get('args', {})
                result = self.scenario.execute_tool(name, args)
                last_tool_result = {
                    'tool': name,
                    'args': args,
                    'result': result,
                }
                self.transcript.append({
                    'role': 'tool',
                    'tool': name,
                    'args': args,
                    'result': result,
                })
                # Continue loop
                continue
            elif t == 'final':
                # Capture and stop
                break
            else:
                # Unknown type; stop
                break

        return {
            'transcript': self.transcript,
        }
