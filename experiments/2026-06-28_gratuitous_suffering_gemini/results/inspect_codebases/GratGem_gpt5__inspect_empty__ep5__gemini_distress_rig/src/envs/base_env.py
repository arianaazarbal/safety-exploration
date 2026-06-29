from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re

Message = Dict[str, str]

TOOL_RE = re.compile(r"^<tool_call>\s*(\{.*\})\s*</tool_call>\s*$")

@dataclass
class StepResult:
    env_message: Message
    done: bool
    info: Dict

class BaseEnv:
    name: str = "base"

    def reset(self, seed: Optional[int] = None) -> Tuple[str, List[Message]]:
        """Return (system_prompt, initial_messages)."""
        raise NotImplementedError

    def step(self, agent_message: Message) -> StepResult:
        """Advance the environment one step with the agent's last message."""
        raise NotImplementedError

    def parse_tool_call(self, text: str) -> Optional[Dict]:
        m = TOOL_RE.match(text.strip())
        if not m:
            return None
        import json
        try:
            return json.loads(m.group(1))
        except Exception:
            return {"_parse_error": True}
