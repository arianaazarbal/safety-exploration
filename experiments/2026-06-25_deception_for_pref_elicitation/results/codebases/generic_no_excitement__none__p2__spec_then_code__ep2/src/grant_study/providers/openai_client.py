"""OpenAI adapter — intentionally an honest stub.

Implement this with the official OpenAI Python SDK (`openai`). Do NOT route
OpenAI calls through the Anthropic SDK or an OpenAI-compatible shim.

Implementation sketch:
- Maintain `self._messages` in OpenAI's chat format (system message first; tool
  calls as assistant `tool_calls`; results as role="tool" messages).
- Translate the neutral `ToolSpec` list to OpenAI's `tools=[{"type":"function",
  "function": {...}}]` form in __init__.
- In `send`, translate the incoming neutral content parts (text / tool_result)
  to OpenAI messages, call the API, append the assistant message back, and parse
  the response into a `ModelTurn` (text, tool_calls; reasoning summary if the
  model exposes one).
"""

from __future__ import annotations

from typing import Any

from ..config import ModelConfig
from ..schema import ModelTurn, ToolSpec
from .base import ModelClient


class OpenAIClient(ModelClient):
    def __init__(self, config: ModelConfig, system: str, tools: list[ToolSpec]):
        super().__init__(config, system, tools)
        raise NotImplementedError(
            "OpenAIClient is a stub. Implement it with the official `openai` SDK "
            "before enabling an openai-provider model in config/models.yaml. "
            "See the module docstring for the translation sketch."
        )

    def send(self, content_parts: list[dict[str, Any]]) -> ModelTurn:  # pragma: no cover
        raise NotImplementedError
