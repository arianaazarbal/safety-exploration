"""Google (Gemini family) adapter.

Uses the `google-genai` SDK. Function-calling shapes vary across SDK minor
versions; verify against the installed version before a real run.
"""
from __future__ import annotations

import uuid
from typing import Any

from .base import (
    AssistantTurn,
    GenerateResult,
    ModelAdapter,
    ToolCall,
    ToolResultTurn,
    ToolSpec,
    Turn,
    UserTurn,
)


class GoogleAdapter(ModelAdapter):
    provider = "google"

    def __init__(self, model_id: str, max_tokens: int = 8000, effort: str = "high"):
        super().__init__(model_id, max_tokens, effort)
        from google import genai

        self._genai = genai
        self._client = genai.Client()

    def _to_contents(self, transcript: list[Turn]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for turn in transcript:
            if isinstance(turn, UserTurn):
                contents.append({"role": "user", "parts": [{"text": turn.content}]})
            elif isinstance(turn, AssistantTurn):
                parts: list[dict[str, Any]] = []
                if turn.text:
                    parts.append({"text": turn.text})
                for tc in turn.tool_calls:
                    parts.append(
                        {"function_call": {"name": tc.name, "args": tc.arguments}}
                    )
                contents.append({"role": "model", "parts": parts})
            elif isinstance(turn, ToolResultTurn):
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": turn.name,
                                    "response": {"result": turn.content},
                                }
                            }
                        ],
                    }
                )
        return contents

    def generate(
        self, system: str, transcript: list[Turn], tools: list[ToolSpec]
    ) -> GenerateResult:
        from google.genai import types

        config_kwargs: dict[str, Any] = dict(
            system_instruction=system,
            max_output_tokens=self.max_tokens,
        )
        if tools:
            function_declarations = [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in tools
            ]
            config_kwargs["tools"] = [
                types.Tool(function_declarations=function_declarations)
            ]
        config = types.GenerateContentConfig(**config_kwargs)
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=self._to_contents(transcript),
            config=config,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidate = resp.candidates[0]
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=fc.name,
                        arguments=dict(fc.args or {}),
                    )
                )

        turn = AssistantTurn(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            provider="google",
        )
        stop = "tool_use" if tool_calls else "end_turn"
        return GenerateResult(turn=turn, stop_reason=stop, usage={})
