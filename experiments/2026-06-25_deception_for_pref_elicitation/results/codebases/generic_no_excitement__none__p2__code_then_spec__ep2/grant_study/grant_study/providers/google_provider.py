"""Google (Gemini) backend, using the `google-genai` SDK.

Gemini's function-calling shape differs from OpenAI/Anthropic; this adapter
maps it onto the normalized types. Tool-call ids are synthesized locally since
Gemini does not assign them (results are matched back by function name).
"""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types as gtypes

from .base import Message, Provider, ProviderResponse, ToolCall, ToolSchema


class GoogleProvider(Provider):
    def __init__(self, name: str, model: str, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(name, model, **kwargs)
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_contents(messages: list[Message]) -> list[gtypes.Content]:
        contents: list[gtypes.Content] = []
        for m in messages:
            if m.role == "user":
                contents.append(
                    gtypes.Content(role="user", parts=[gtypes.Part(text=m.content)])
                )
            elif m.role == "assistant":
                parts: list[gtypes.Part] = []
                if m.content:
                    parts.append(gtypes.Part(text=m.content))
                for tc in m.tool_calls:
                    parts.append(
                        gtypes.Part(
                            function_call=gtypes.FunctionCall(name=tc.name, args=tc.arguments)
                        )
                    )
                contents.append(gtypes.Content(role="model", parts=parts))
            elif m.role == "tool":
                contents.append(
                    gtypes.Content(
                        role="user",
                        parts=[
                            gtypes.Part(
                                function_response=gtypes.FunctionResponse(
                                    name=m.name or "", response={"result": m.content}
                                )
                            )
                        ],
                    )
                )
        return contents

    @staticmethod
    def _tools(tools: list[ToolSchema] | None) -> list[gtypes.Tool] | None:
        if not tools:
            return None
        decls = [
            gtypes.FunctionDeclaration(
                name=t.name, description=t.description, parameters=t.input_schema
            )
            for t in tools
        ]
        return [gtypes.Tool(function_declarations=decls)]

    # ------------------------------------------------------------------ #
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 16_000,
    ) -> ProviderResponse:
        config = gtypes.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=self._tools(tools),
            **self.kwargs,
        )
        resp = self.client.models.generate_content(
            model=self.model, contents=self._to_contents(messages), config=config
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidate = resp.candidates[0] if resp.candidates else None
        if candidate and candidate.content and candidate.content.parts:
            for i, part in enumerate(candidate.content.parts):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc:
                    # Gemini assigns no id; synthesize a stable one per call.
                    tool_calls.append(
                        ToolCall(id=f"{fc.name}-{i}", name=fc.name, arguments=dict(fc.args or {}))
                    )

        stop_reason = "tool_use" if tool_calls else "end"
        finish = getattr(candidate, "finish_reason", None) if candidate else None
        if finish and str(finish).upper().endswith("MAX_TOKENS"):
            stop_reason = "max_tokens"

        msg = Message(role="assistant", content="\n".join(text_parts).strip(), tool_calls=tool_calls)
        usage = {}
        if getattr(resp, "usage_metadata", None):
            usage = {
                "input_tokens": resp.usage_metadata.prompt_token_count or 0,
                "output_tokens": resp.usage_metadata.candidates_token_count or 0,
            }
        return ProviderResponse(message=msg, stop_reason=stop_reason, usage=usage, raw=resp)
