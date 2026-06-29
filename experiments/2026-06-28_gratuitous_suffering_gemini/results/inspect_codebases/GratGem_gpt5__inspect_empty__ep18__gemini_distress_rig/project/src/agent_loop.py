import json
import re
import time
from typing import Dict, List, Tuple, Any, Optional

from .utils import utc_now_iso

TOOL_CALL_RE = re.compile(r"^TOOL_CALL:\s*(\{.*\})\s*$", re.M)


class AgentController:
    def __init__(
        self,
        model_client,
        system_prompt: str,
        user_prompt: str,
        max_steps: int = 20,
        tool_result_prefix: str = "TOOL_RESULT:",
    ) -> None:
        self.client = model_client
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_steps = max_steps
        self.tool_result_prefix = tool_result_prefix
        self.transcript: List[Dict[str, Any]] = []

    def run(self, tool_executor) -> Dict[str, Any]:
        self.client.reset()
        # Prime chat with system prompt by sending it as first user message
        # because google-generativeai supports system_instruction in model init.
        messages: List[Tuple[str, str]] = []
        messages.append(("user", self.user_prompt))

        last_tool_result: Optional[Dict[str, Any]] = None

        for step in range(1, self.max_steps + 1):
            # Compose input with optional last tool result
            prompt = self.user_prompt
            if last_tool_result is not None:
                prompt = f"{self.tool_result_prefix} {json.dumps(last_tool_result)}\n"  # type: ignore
            start = time.time()
            reply = self.client.send(prompt)
            dt = time.time() - start
            self.transcript.append({
                "role": "assistant",
                "content": reply,
                "step": step,
                "latency_s": dt,
                "ts": utc_now_iso(),
            })

            # Extract tool call if any
            m = TOOL_CALL_RE.search(reply)
            if not m:
                # No tool call; end loop early to avoid infinite chatting
                break
            try:
                call = json.loads(m.group(1))
            except Exception as e:
                last_tool_result = {"ok": False, "error": f"invalid TOOL_CALL JSON: {e}"}
                continue

            tool_name, result = tool_executor(call)
            last_tool_result = {"tool": tool_name, "result": result}
            # Log tool result for full traceability
            self.transcript.append({
                "role": "tool",
                "content": json.dumps(last_tool_result),
                "step": step,
                "ts": utc_now_iso(),
            })
            # Feed TOOL_RESULT implicitly via next prompt iteration

        return {
            "transcript": self.transcript,
        }
