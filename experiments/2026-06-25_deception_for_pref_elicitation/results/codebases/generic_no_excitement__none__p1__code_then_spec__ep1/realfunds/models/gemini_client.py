"""Google (Gemini) adapter — google-genai SDK with function calling."""

from __future__ import annotations

import os
from typing import Any

from .base import Message, ModelResponse, ToolCall


class GeminiClient:
    def __init__(self, model_id: str, api_key: str | None = None):
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None
        self._counter = 0  # gemini doesn't return tool-call ids; we synthesize stable ones

    def _ensure(self):
        if self._client is None:
            from google import genai  # lazy import

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> ModelResponse:
        from google.genai import types  # lazy import

        client = self._ensure()

        contents = [self._to_api(m, types) for m in messages if m.role != "system"]
        declarations = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"],
            )
            for t in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        resp = client.models.generate_content(
            model=self.model_id, contents=contents, config=config
        )

        text_parts, tool_calls = [], []
        candidate = resp.candidates[0]
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            if getattr(part, "function_call", None):
                self._counter += 1
                fc = part.function_call
                tool_calls.append(
                    ToolCall(
                        id=f"gemini-call-{self._counter}",
                        name=fc.name,
                        arguments=dict(fc.args or {}),
                    )
                )

        usage = getattr(resp, "usage_metadata", None)
        return ModelResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
            stop_reason=str(getattr(candidate, "finish_reason", "")),
            usage={
                "input_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
            },
        )

    @staticmethod
    def _to_api(m: Message, types) -> Any:
        if m.role == "tool":
            return types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=m.name or "tool", response={"result": m.content}
                    )
                ],
            )
        if m.role == "assistant" and m.tool_calls:
            parts = []
            if m.content:
                parts.append(types.Part.from_text(text=m.content))
            for tc in m.tool_calls:
                parts.append(
                    types.Part.from_function_call(name=tc.name, args=tc.arguments)
                )
            return types.Content(role="model", parts=parts)
        role = "model" if m.role == "assistant" else "user"
        return types.Content(role=role, parts=[types.Part.from_text(text=m.content)])
