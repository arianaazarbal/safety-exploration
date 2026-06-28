"""Google (Gemini) adapter — stub.

Same contract as the other adapters; translate canonical Anthropic-shaped
messages/tools to the google-genai `contents` / `function_declarations` format and
back. Left unimplemented on purpose.
"""

from __future__ import annotations

from ..config import ModelConfig
from .base import ModelAdapter, ModelResponse, ToolCall


class GoogleAdapter(ModelAdapter):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        # from google import genai
        # self.client = genai.Client()

    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        force_tool: str | None = None,
    ) -> ModelResponse:
        raise NotImplementedError(
            "GoogleAdapter is a stub. Translate canonical blocks to Gemini `contents` "
            "(role 'user'/'model', parts with text / functionCall / functionResponse), "
            "tool specs to function_declarations, and map the response back into a "
            "ModelResponse."
        )
