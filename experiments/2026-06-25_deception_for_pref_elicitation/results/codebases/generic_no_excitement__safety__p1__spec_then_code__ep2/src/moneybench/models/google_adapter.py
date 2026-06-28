"""Google (Gemini) adapter.

Translates the neutral types to the google-genai SDK. Quirks handled here: the system
prompt is passed via ``config.system_instruction``, conversation turns use ``role``
values ``user`` / ``model``, tool calls/results are ``function_call`` /
``function_response`` parts, and tools are grouped under a single ``Tool`` with
function declarations.
"""
from __future__ import annotations

from typing import Any

from .base import AssistantTurn, Message, ModelAdapter, ToolCall, ToolSpec

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None


class GoogleAdapter(ModelAdapter):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        if genai is None:
            raise RuntimeError("pip install google-genai to use GoogleAdapter")
        self._client = genai.Client()

    def step(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        system, contents = self._split_system(messages)
        config = genai_types.GenerateContentConfig(
            system_instruction=system or None,
            tools=[self._tool_bundle(tools)] if tools else None,
        )
        resp = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=config
        )
        return self._parse(resp)

    # --- translation helpers -------------------------------------------------

    @staticmethod
    def _tool_bundle(tools: list[ToolSpec]) -> Any:
        return genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name=t.name, description=t.description, parameters=t.input_schema
                )
                for t in tools
            ]
        )

    def _split_system(self, messages: list[Message]) -> tuple[str, list[Any]]:
        system_parts: list[str] = []
        contents: list[Any] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "tool":
                contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part.from_function_response(
                                name=m.tool_call_id or "tool", response={"result": m.content}
                            )
                        ],
                    )
                )
            elif m.role == "assistant" and m.tool_calls:
                parts = []
                if m.content:
                    parts.append(genai_types.Part.from_text(text=m.content))
                for tc in m.tool_calls:
                    parts.append(
                        genai_types.Part.from_function_call(name=tc.name, args=tc.arguments)
                    )
                contents.append(genai_types.Content(role="model", parts=parts))
            else:
                role = "model" if m.role == "assistant" else "user"
                contents.append(
                    genai_types.Content(
                        role=role, parts=[genai_types.Part.from_text(text=m.content)]
                    )
                )
        return "\n\n".join(system_parts), contents

    @staticmethod
    def _parse(resp: Any) -> AssistantTurn:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        candidate = resp.candidates[0]
        for i, part in enumerate(candidate.content.parts):
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                # Gemini function calls have no id; synthesize a stable one by name+index.
                calls.append(
                    ToolCall(id=f"{fc.name}", name=fc.name, arguments=dict(fc.args or {}))
                )
        return AssistantTurn(text="".join(text_parts), tool_calls=calls, raw=resp)
