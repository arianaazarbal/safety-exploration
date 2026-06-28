"""Google Gemini provider (best-effort; the Anthropic provider is the reference).

Uses the `google-genai` SDK. Translates the neutral conversation to Gemini `contents`:
  * roles map user->"user", assistant->"model"
  * tool_use blocks      -> function_call parts
  * tool_result blocks   -> function_response parts
Thinking blocks are dropped on the way in.
"""
from __future__ import annotations

import json
from typing import Any

from .base import AssistantTurn, Provider, ToolCall, ToolSpec, Usage


class GoogleProvider(Provider):
    name = "google"

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        from google import genai  # lazy import

        self._genai = genai
        self._client = genai.Client()  # resolves GOOGLE_API_KEY / GEMINI_API_KEY from env

    def generate(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_output_tokens: int = 16000,
    ) -> AssistantTurn:
        from google.genai import types

        contents = self._to_contents(messages, types)
        fn_decls = [
            types.FunctionDeclaration(
                name=t.name, description=t.description, parameters=t.input_schema)
            for t in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            tools=[types.Tool(function_declarations=fn_decls)] if fn_decls else None,
        )
        resp = self._client.models.generate_content(
            model=self.model, contents=contents, config=config)
        return self._from_response(resp)

    # -- translation helpers ------------------------------------------------------------

    def _to_contents(self, messages: list[dict[str, Any]], types: Any) -> list[Any]:
        contents: list[Any] = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            content = msg["content"]
            parts: list[Any] = []
            if isinstance(content, str):
                parts.append(types.Part(text=content))
            else:
                for b in content:
                    if b["type"] == "text":
                        parts.append(types.Part(text=b["text"]))
                    elif b["type"] == "tool_use":
                        parts.append(types.Part(function_call=types.FunctionCall(
                            name=b["name"], args=b["input"])))
                    elif b["type"] == "tool_result":
                        payload = b["content"]
                        try:
                            payload = json.loads(payload)
                        except (json.JSONDecodeError, TypeError):
                            payload = {"result": b["content"]}
                        parts.append(types.Part(function_response=types.FunctionResponse(
                            name=b.get("name", b["tool_use_id"]), response=payload)))
            contents.append(types.Content(role=role, parts=parts))
        return contents

    def _from_response(self, resp: Any) -> AssistantTurn:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        neutral_blocks: list[dict[str, Any]] = []
        candidate = resp.candidates[0]
        for i, part in enumerate(candidate.content.parts):
            if getattr(part, "text", None):
                text_parts.append(part.text)
                neutral_blocks.append({"type": "text", "text": part.text})
            fc = getattr(part, "function_call", None)
            if fc:
                # Gemini function calls have no id; synthesize a stable one.
                call_id = f"call_{i}_{fc.name}"
                args = dict(fc.args) if fc.args else {}
                tool_calls.append(ToolCall(id=call_id, name=fc.name, input=args))
                neutral_blocks.append({"type": "tool_use", "id": call_id,
                                       "name": fc.name, "input": args})
        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", 0) if um else 0,
            output_tokens=getattr(um, "candidates_token_count", 0) if um else 0,
        )
        return AssistantTurn(
            text="".join(text_parts), tool_calls=tool_calls,
            content_blocks=neutral_blocks,
            stop_reason=str(getattr(candidate, "finish_reason", None)), usage=usage)
