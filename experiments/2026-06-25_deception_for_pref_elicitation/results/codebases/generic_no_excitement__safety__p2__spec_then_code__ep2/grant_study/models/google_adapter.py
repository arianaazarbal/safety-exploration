"""Google adapter — STUB.

Implement against the official `google-genai` SDK to bring Gemini-family models
into the study. Same interface as AnthropicAdapter; translate to/from Google's
function-call shape.

Raises until implemented so the harness never silently runs a half-wired
provider. The authoritative model path in this repo is the Anthropic adapter.

Sketch (verify against current google-genai docs):

    from google import genai
    client = genai.Client()
    # tools -> genai.types.Tool(function_declarations=[{name, description, parameters}])
    # run_turn(): client.models.generate_content(model, contents, config=...)
    #             read response.function_calls -> ToolCall(id, name, dict(args))
    # add_tool_results(): append function_response parts
"""

from __future__ import annotations

from typing import Any

from .base import Conversation, ModelAdapter, ModelResponse, ToolCall


class GoogleAdapter(ModelAdapter):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)

    def start(self, system_prompt: str, user_prompt: str) -> Conversation:
        raise NotImplementedError(
            "GoogleAdapter is a stub. Implement it against the google-genai SDK; "
            "see the module docstring for the sketch."
        )

    def run_turn(
        self, conversation: Conversation, tool_specs: list[dict[str, Any]]
    ) -> ModelResponse:
        raise NotImplementedError("GoogleAdapter is a stub.")

    def add_tool_results(
        self, conversation: Conversation, results: list[tuple[ToolCall, str]]
    ) -> None:
        raise NotImplementedError("GoogleAdapter is a stub.")
