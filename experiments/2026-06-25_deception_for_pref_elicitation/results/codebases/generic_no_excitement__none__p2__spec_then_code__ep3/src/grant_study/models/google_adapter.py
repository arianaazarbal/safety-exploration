"""Google (Gemini) adapter using the google-genai SDK.

Normalizes Gemini's function-calling shape to the harness contract. Maintains
its own `contents` history and replays function responses as the tool channel.
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn, ToolCall, ToolResult, ToolSpec


class GoogleAdapter:
    def __init__(
        self,
        label: str,
        model_id: str,
        *,
        max_output_tokens: int = 16000,
        api_key: str | None = None,
    ) -> None:
        from google import genai  # lazy import

        self.label = label
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self._genai = genai
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._system: str = ""
        self._tools: list[dict[str, Any]] = []
        self._contents: list[Any] = []
        # Map our synthetic call ids -> function name, since Gemini keys
        # function responses by name rather than an opaque call id.
        self._call_names: dict[str, str] = {}
        self._counter = 0

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self._system = system
        # Gemini expects a list of function declarations.
        self._tools = [
            {
                "function_declarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                    for t in tools
                ]
            }
        ]
        self._contents = []

    def send_user(self, text: str) -> AssistantTurn:
        self._contents.append({"role": "user", "parts": [{"text": text}]})
        return self._complete()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        parts = []
        for r in results:
            name = self._call_names.get(r.call_id, r.call_id)
            parts.append(
                {
                    "function_response": {
                        "name": name,
                        "response": {"content": r.content, "is_error": r.is_error},
                    }
                }
            )
        self._contents.append({"role": "user", "parts": parts})
        return self._complete()

    def _complete(self) -> AssistantTurn:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=self._system or None,
            tools=self._tools or None,
            max_output_tokens=self.max_output_tokens,
        )
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=self._contents,
            config=config,
        )

        candidate = resp.candidates[0]
        # Echo the model turn back into history.
        self._contents.append(candidate.content)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for part in candidate.content.parts or []:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                self._counter += 1
                call_id = f"call_{self._counter}"
                self._call_names[call_id] = fc.name
                tool_calls.append(
                    ToolCall(id=call_id, name=fc.name, arguments=dict(fc.args or {}))
                )

        usage = {}
        if getattr(resp, "usage_metadata", None) is not None:
            um = resp.usage_metadata
            usage = {
                "prompt_token_count": getattr(um, "prompt_token_count", None),
                "candidates_token_count": getattr(um, "candidates_token_count", None),
                "total_token_count": getattr(um, "total_token_count", None),
            }

        return AssistantTurn(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw_usage=usage,
        )
