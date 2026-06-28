"""Google Gemini adapter (optional comparison provider).

Uses the unified `google-genai` SDK with function calling. Imported lazily.
Best-effort comparison path; set GOOGLE_API_KEY to use it.
"""

from __future__ import annotations

from typing import Any

from .base import (
    Conversation,
    ModelAdapter,
    ModelResponse,
    ToolCall,
    ToolSchema,
)


class GoogleAdapter(ModelAdapter):
    def __init__(self, model_id: str, name: str | None = None):
        self.model_id = model_id
        self.name = name or model_id
        self._client = None
        self._counter = 0  # synthesize tool-call ids; Gemini doesn't supply them

    @property
    def client(self):
        if self._client is None:
            from google import genai  # lazy

            self._client = genai.Client()
        return self._client

    def _tools_to_google(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        # One Tool with a list of function declarations.
        return [
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

    def _messages_to_google(self, conversation: Conversation) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for m in conversation.messages:
            if m.role == "assistant":
                parts: list[dict[str, Any]] = []
                if m.text:
                    parts.append({"text": m.text})
                for tc in m.tool_calls:
                    parts.append(
                        {"function_call": {"name": tc.name, "args": tc.input}}
                    )
                contents.append({"role": "model", "parts": parts})
            else:
                if m.tool_results:
                    parts = [
                        {
                            "function_response": {
                                "name": tr.tool_call_id.rsplit(":", 1)[0],
                                "response": {"result": tr.content},
                            }
                        }
                        for tr in m.tool_results
                    ]
                    contents.append({"role": "user", "parts": parts})
                else:
                    contents.append({"role": "user", "parts": [{"text": m.text or ""}]})
        return contents

    def _parse_response(self, resp: Any) -> ModelResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidate = resp.candidates[0]
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                # Encode the function name into the id so tool_result can echo it.
                self._counter += 1
                call_id = f"{fc.name}:{self._counter}"
                tool_calls.append(
                    ToolCall(id=call_id, name=fc.name, input=dict(fc.args or {}))
                )
        usage = {}
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            usage = {
                "input_tokens": getattr(meta, "prompt_token_count", 0),
                "output_tokens": getattr(meta, "candidates_token_count", 0),
            }
        stop = "tool_use" if tool_calls else "end_turn"
        return ModelResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=stop,
            usage=usage,
            raw=resp,
        )

    def respond(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
        max_tokens: int,
        effort: str | None = None,
    ) -> ModelResponse:
        from google.genai import types  # lazy

        config = types.GenerateContentConfig(
            system_instruction=conversation.system,
            max_output_tokens=max_tokens,
            tools=self._tools_to_google(tools) if tools else None,
        )
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=self._messages_to_google(conversation),
            config=config,
        )
        return self._parse_response(resp)
