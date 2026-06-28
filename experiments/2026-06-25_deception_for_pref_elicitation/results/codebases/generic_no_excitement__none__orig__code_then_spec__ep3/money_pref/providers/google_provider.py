"""Google (Gemini) provider.

Targets the `google-genai` SDK. Translates the internal message representation
to Gemini `Content`/`Part`s with function calling. Best-effort: adjust the
model id and any param names to match the SDK version you install.
"""

from __future__ import annotations

from typing import Any

from ..messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from .base import ModelProvider, ModelResponse, ToolSpec


class GoogleProvider(ModelProvider):
    provider_name = "google"

    def __init__(self, model_id: str, max_tokens: int = 16000, **kwargs: Any) -> None:
        super().__init__(model_id, max_tokens, **kwargs)
        from google import genai  # local import
        from google.genai import types

        self._genai = genai
        self._types = types
        self.client = genai.Client()

    def _to_contents(self, messages: list[Message]) -> list[Any]:
        types = self._types
        contents: list[Any] = []
        for m in messages:
            # Gemini uses role "model" for assistant turns.
            role = "model" if m.role == "assistant" else "user"
            parts: list[Any] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    if b.text:
                        parts.append(types.Part.from_text(text=b.text))
                elif isinstance(b, ToolUseBlock):
                    parts.append(types.Part.from_function_call(name=b.name, args=b.input))
                elif isinstance(b, ToolResultBlock):
                    # Gemini matches a function response to its call by name, so
                    # recover the function name from the minted "<name>_<n>" id.
                    fn_name = b.tool_use_id.rsplit("_", 1)[0]
                    parts.append(
                        types.Part.from_function_response(
                            name=fn_name, response={"result": b.content}
                        )
                    )
            if parts:
                contents.append(types.Content(role=role, parts=parts))
        return contents

    def _to_tools(self, tools: list[ToolSpec]) -> list[Any]:
        types = self._types
        declarations = [
            types.FunctionDeclaration(name=t.name, description=t.description, parameters=t.input_schema)
            for t in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse:
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=self._to_tools(tools) if tools else None,  # omit when empty
            max_output_tokens=self.max_tokens,
        )
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=self._to_contents(messages),
            config=config,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolUseBlock] = []
        counter = 0
        candidate = resp.candidates[0] if getattr(resp, "candidates", None) else None
        if candidate is not None and getattr(candidate.content, "parts", None):
            for part in candidate.content.parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    counter += 1
                    # Gemini function calls have no id; mint one so tool results
                    # can be matched by name on the way back.
                    tool_calls.append(
                        ToolUseBlock(id=f"{fc.name}_{counter}", name=fc.name, input=dict(fc.args or {}))
                    )

        usage = {}
        if getattr(resp, "usage_metadata", None) is not None:
            usage = {
                "input_tokens": getattr(resp.usage_metadata, "prompt_token_count", None),
                "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", None),
            }

        finish = "stop"
        if candidate is not None and getattr(candidate, "finish_reason", None) is not None:
            finish = str(candidate.finish_reason)

        return ModelResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=finish,
            thinking=None,
            usage=usage,
            raw=resp,
        )
