"""Google (Gemini) adapter — intentionally an honest stub.

Implement this with the official Google generative-AI SDK (`google-genai`). Do
NOT route Google calls through another provider's SDK.

Implementation sketch:
- Maintain conversation history in the SDK's `contents` format (roles "user" /
  "model"); tool calls are `function_call` parts, results are `function_response`
  parts.
- Translate the neutral `ToolSpec` list to the SDK's function-declaration form.
- In `send`, translate neutral content parts to `contents`, call the model,
  append the response, and parse into a `ModelTurn`.
"""

from __future__ import annotations

from typing import Any

from ..config import ModelConfig
from ..schema import ModelTurn, ToolSpec
from .base import ModelClient


class GoogleClient(ModelClient):
    def __init__(self, config: ModelConfig, system: str, tools: list[ToolSpec]):
        super().__init__(config, system, tools)
        raise NotImplementedError(
            "GoogleClient is a stub. Implement it with the official `google-genai` "
            "SDK before enabling a google-provider model in config/models.yaml. "
            "See the module docstring for the translation sketch."
        )

    def send(self, content_parts: list[dict[str, Any]]) -> ModelTurn:  # pragma: no cover
        raise NotImplementedError
